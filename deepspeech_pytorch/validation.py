from abc import ABC, abstractmethod

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from deepspeech_pytorch.decoder import Decoder, GreedyDecoder

from pytorch_lightning.metrics import Metric
import Levenshtein as Lev

from deepspeech_pytorch.average import get_res, complet_csv, get_res_gauss
from deepspeech_pytorch.metric import get_correlation_values
from deepspeech_pytorch.configs.train_config import  DTWDataConfig


class ErrorRate(Metric, ABC):
    def __init__(
        self,
        decoder: Decoder,
        target_decoder: GreedyDecoder,
        save_output: bool = False,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output

    @abstractmethod
    def calculate_metric(self, transcript, reference):
        raise NotImplementedError

    def update(
        self,
        preds: torch.Tensor,
        preds_sizes: torch.Tensor,
        targets: torch.Tensor,
        target_sizes: torch.Tensor,
    ):
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size
        decoded_output, _ = self.decoder.decode(preds, preds_sizes)
        target_strings = self.target_decoder.convert_to_strings(split_targets)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            self.calculate_metric(transcript=transcript, reference=reference)


class CharErrorRate(ErrorRate):
    def __init__(
        self,
        decoder: Decoder,
        target_decoder: GreedyDecoder,
        save_output: bool = False,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("cer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_chars", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        cer_inst = self.cer_calc(transcript, reference)
        self.cer += cer_inst
        self.n_chars += len(reference.replace(" ", ""))

    def compute(self):
        cer = float(self.cer) / self.n_chars
        return cer.item() * 100

    def cer_calc(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(
            " ", ""
        ), s2.replace(" ", "")
        return Lev.distance(s1, s2)


class WordErrorRate(ErrorRate):
    def __init__(
        self,
        decoder: Decoder,
        target_decoder: GreedyDecoder,
        save_output: bool = False,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("wer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_tokens += len(reference.split())

    def compute(self):
        wer = float(self.wer) / self.n_tokens
        return wer.item() * 100

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance("".join(w1), "".join(w2))


@torch.no_grad()
def run_evaluation(
    test_loader,
    model,
    decoder: Decoder,
    device: torch.device,
    target_decoder: Decoder,
    precision: int,
):
    model.eval()
    wer = WordErrorRate(decoder=decoder, target_decoder=target_decoder)
    cer = CharErrorRate(decoder=decoder, target_decoder=target_decoder)
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        with autocast(enabled=precision == 16):
            out, output_sizes = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_sizes)
        wer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes,
        )
        cer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes,
        )
    return wer.compute(), cer.compute()


@torch.no_grad()
def run_evaluationdtw(
    human_csv, test_loader, device: torch.device, model, precision: int, representation
):
    model.eval()
    id_triplets = []
    delta_values = []
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data = batch
        TGT, OTH, X = data[0], data[1], data[2]
        TGT, OTH, X = TGT.to(device), OTH.to(device), X.to(device)
        id_triplets.append(data[3]) # we get the triplet id

        with autocast(enabled=precision == 16):
            TGT_output, OTH_output, X_output = model(TGT, OTH, X)

        try:
            if representation == "gauss":
                a = get_res_gauss(
                    TGT_output.cpu().numpy(),
                    OTH_output.cpu().numpy(),
                    X_output.cpu().numpy(),
                )
            if representation == "normal":
                a = get_res(
                    TGT_output.cpu().numpy(),
                    OTH_output.cpu().numpy(),
                    X_output.cpu().numpy(),
                )
        except:
            OTH_output = torch.add(OTH_output, 10e-10)
            X_output = torch.add(X_output, 10e-10)
            TGT_output = torch.add(TGT_output, 10e-10)

            if representation == "gauss":
                a = get_res_gauss(
                    TGT_output.cpu().numpy(),
                    OTH_output.cpu().numpy(),
                    X_output.cpu().numpy(),
                )
            if representation == "normal":
                a = get_res(
                    TGT_output.cpu().numpy(),
                    OTH_output.cpu().numpy(),
                    X_output.cpu().numpy(),
                )
        delta_values.append(a) # we get the corresponding delta value

    #print(delta_values, id_triplets)
    id_triplets2 = [id_triplets[x][0] for x in range(len(id_triplets))]
    df = complet_csv(human_csv, delta_values, id_triplets2)

    print(df[:10])
    spearman = get_correlation_values(
        df["user_ans"], df["delta_values"], mode="spearman"
    )
    pearson = get_correlation_values(df["user_ans"], df["delta_values"], mode="pearson")

    pos = [x for x in delta_values if x >= 0]
    delta_positivity = (len(pos) / len(delta_values)) * 100

    return spearman, pearson, delta_positivity
