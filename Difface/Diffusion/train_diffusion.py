import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import torch.nn as nn
import click


def pad_gather_reduce(model, x, method="mean"):
    """
    Gather a value or tensor across all processes and reduce it.

    Args:
        model: trainer object that carries `accelerator` and `device`
        x: a number or torch tensor to reduce
        method: one of ["mean", "sum", "max", "min"]

    Returns:
        Reduced tensor, or None if gather result is empty.
    """
    assert method in ["mean", "sum", "max", "min"], \
        "This function has limited capabilities [sum, mean, max, min]"
    assert x is not None, "Cannot reduce a None type object"

    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], dtype=torch.float32, device=model.device)
    else:
        x = x.to(model.device)

    # flatten for safer cross-process gathering
    x = x.reshape(-1)

    padded_x = model.accelerator.pad_across_processes(x, dim=0)
    gathered_x = model.accelerator.gather(padded_x)

    # Do NOT mask out zeros. Zero can be a valid cosine similarity.
    if gathered_x.numel() == 0:
        click.secho(
            "The call to this method resulted in an empty tensor after gather.",
            fg="red",
        )
        return None

    if method == "mean":
        return torch.mean(gathered_x)
    elif method == "sum":
        return torch.sum(gathered_x)
    elif method == "max":
        return torch.max(gathered_x)
    elif method == "min":
        return torch.min(gathered_x)


def train(decoder, model, train_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for train_dataset in train_loader:
        snp, face = train_dataset
        images = face.to(device)
        text = snp.to(device)

        loss = model(text, images)
        model.update()

        total_loss += float(loss)
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


@torch.no_grad()
def test(decoder, model, loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for data in loader:
        snp, face = data
        images = face.to(device)
        text = snp.to(device)

        loss = model(text, images)

        total_loss += float(loss)
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


@torch.no_grad()
def out(decoder, model, loader, device):
    raise NotImplementedError(
        "The original SHAP-based `out()` path is incomplete and unsafe to run. "
        "Please rewrite it separately if SHAP analysis is really needed."
    )


def run(decoder, clip, model, train_loader, test_loader, epochs, writer, device):
    train_losses, test_losses = [], []

    out_dir = '/share/home/jiaomingqi/diffusion5/out/checkpoint/'
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(0, epochs):
        if epoch % 10 == 0:
            test_loss = test(decoder, model, test_loader, device)
            test_losses.append(test_loss)
            print(epoch, 'test_loss', test_loss)

            orig_sim, pred_sim, pred_img_sim = report_cosine_sims(
                decoder, clip, model, test_loader, device
            )
            print(
                epoch,
                'orig_sim', orig_sim,
                'pred_sim', pred_sim,
                'pred_img_sim', pred_img_sim
            )

        train_loss = train(decoder, model, train_loader, device)
        train_losses.append(train_loss)
        print(epoch, 'train_loss', train_loss)

        if epoch % 10 == 0:
            model.save(out_dir + str(epoch) + '_checkpoint')

    return train_losses, test_losses


cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def _safe_l2_normalize(x, eps=1e-12):
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def report_cosine_sims(decoder, clip, model, loader, device):
    model.eval()

    orig_vals = []
    pred_vals = []
    pred_img_vals = []

    for data in loader:
        snp, face = data
        images = face.to(device)
        text = snp.to(device)

        test_image_embeddings = clip.embed_image(images)
        text_embed = clip.embed_text(text)
        predicted_image_embeddings = model.sample(text)

        # safer normalization
        text_embed = _safe_l2_normalize(text_embed)
        test_image_embeddings = _safe_l2_normalize(test_image_embeddings)
        predicted_image_embeddings = _safe_l2_normalize(predicted_image_embeddings)

        orig_batch = cos(text_embed, test_image_embeddings)
        pred_batch = cos(text_embed, predicted_image_embeddings)
        pred_img_batch = cos(test_image_embeddings, predicted_image_embeddings)

        orig_vals.append(orig_batch.detach())
        pred_vals.append(pred_batch.detach())
        pred_img_vals.append(pred_img_batch.detach())

    if len(orig_vals) == 0:
        return None, None, None

    orig_vals = torch.cat(orig_vals, dim=0)
    pred_vals = torch.cat(pred_vals, dim=0)
    pred_img_vals = torch.cat(pred_img_vals, dim=0)

    orig_sim = pad_gather_reduce(model, orig_vals, method="mean")
    pred_sim = pad_gather_reduce(model, pred_vals, method="mean")
    pred_img_sim = pad_gather_reduce(model, pred_img_vals, method="mean")

    return orig_sim, pred_sim, pred_img_sim
        
