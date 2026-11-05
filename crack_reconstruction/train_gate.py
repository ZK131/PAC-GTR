"""Original L515 gate training with explicit input/output paths. No data are bundled."""
import argparse, json, random, sys, time, csv
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from crack_reconstruction.image.model_v3_context import DUTRContextV3
from crack_reconstruction.image.losses import soft_dice_loss

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared-root',required=True)
    p.add_argument('--context-checkpoint',required=True)
    p.add_argument('--ce-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    SEED = 20260928
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEV = torch.device(args.device)
    V1 = Path(args.prepared_root)
    CTX = Path(args.context_checkpoint).parent
    CECK = Path(args.ce_checkpoint)
    OUT = Path(args.output)
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError("Output directory must be new")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / "train.log"
    CSV = OUT / "metrics.csv"
    BEST = OUT / "best.pt"
    SUMMARY = OUT / "summary.json"
    LOG.write_text("", encoding="utf-8")

    def log(s: str):
        z = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}"
        print(z, flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(z + "\n")

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = Block(3,16)
            self.e2 = Block(16,32)
            self.e3 = Block(32,64)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(64,32,2,2)
            self.u1 = nn.ConvTranspose2d(32,16,2,2)
            self.d2 = Block(64,32)
            self.d1 = Block(32,16)
            self.out = nn.Conv2d(16,1,1)
        def forward(self, x):
            a = self.e1(x)
            b = self.e2(self.pool(a))
            c = self.e3(self.pool(b))
            z = self.d2(torch.cat([self.u2(c), b], 1))
            z = self.d1(torch.cat([self.u1(z), a], 1))
            return self.out(z)

    class ContextCEGate(nn.Module):
        """Conservative 2-D residual calibrator around frozen Context2D.

        head starts at 0, so initialization is exact Context2D; a small nonzero scale keeps gradients alive.
        """
        def __init__(self, width=24, max_corr=1.0):
            super().__init__()
            self.max_corr = float(max_corr)
            self.stem = nn.Sequential(
                nn.Conv2d(8, width, 3, padding=1, bias=False),
                nn.GroupNorm(6, width),
                nn.GELU(),
            )
            self.d1 = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
            self.d3 = nn.Conv2d(width, width, 3, padding=3, dilation=3, groups=width, bias=False)
            self.mix = nn.Sequential(
                nn.Conv2d(width*2, width, 1, bias=False),
                nn.GroupNorm(6, width),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.GroupNorm(6, width),
                nn.GELU(),
            )
            self.head = nn.Conv2d(width, 1, 1)
            self.gate = nn.Conv2d(width, 1, 1)
            self.scale = nn.Parameter(torch.tensor(0.05))
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, p, q, c, rgb):
            sup = F.avg_pool2d(p, 7, 1, 3)
            unc = 1.0 - (2.0*q - 1.0).abs()
            qp = q - p
            cq = c - q
            gray = rgb.mean(1, keepdim=True)
            dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            dx = F.pad(dx, (0,1,0,0))
            dy = F.pad(dy, (0,0,0,1))
            edge = (dx + dy).clamp(0,1)
            x = torch.cat([p, q, c, qp, cq, sup, unc, edge], 1)
            z = self.stem(x)
            z = self.mix(torch.cat([self.d1(z), self.d3(z)], 1))
            local_gate = torch.sigmoid(self.gate(z)) * (0.35 + 0.65*unc)
            corr = self.max_corr * torch.tanh(self.head(z)) * local_gate * torch.tanh(self.scale)
            ql = torch.logit(q.clamp(1e-4, 1-1e-4))
            return ql + corr

    items = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))["items"]
    P = V1 / "omni_fold0_probs256"

    class DS(Dataset):
        def __init__(self, split, aug=False):
            self.a = [x for x in items if x["split"] == split]
            self.aug = aug
        def __len__(self):
            return len(self.a)
        def __getitem__(self, i):
            x = self.a[i]
            idx = int(x["id"])
            im = cv2.cvtColor(cv2.imread(x["rgb"]), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (256,256), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
            y = (cv2.imread(x["mask256"], 0) > 127).astype(np.float32)[None]
            p = np.load(P / f"{idx:04d}.npy").astype(np.float32)[None]
            rgb = torch.from_numpy(im.transpose(2,0,1).copy()).float()
            if self.aug:
                if random.random() < 0.75:
                    rgb = (rgb*random.uniform(0.82,1.18) + random.uniform(-0.06,0.06)).clamp(0,1)
                if random.random() < 0.30:
                    rgb = (rgb + torch.randn_like(rgb)*random.uniform(0,0.018)).clamp(0,1)
                if random.random() < 0.22:
                    k = random.choice([3,5])
                    rgb = F.avg_pool2d(rgb[None], k, 1, k//2)[0]
            return rgb, torch.from_numpy(p), torch.from_numpy(y)

    context = DUTRContextV3(base_channels=32, ctx_channels=24, max_context=1.25).to(DEV)
    ctxck = torch.load(Path(args.context_checkpoint), map_location=DEV, weights_only=False)
    context.load_state_dict(ctxck["model"])
    context.eval()
    for x in context.parameters():
        x.requires_grad = False

    ce = TinyUNet().to(DEV)
    ce.load_state_dict(torch.load(CECK, map_location=DEV, weights_only=False)["model"])
    ce.eval()
    for x in ce.parameters():
        x.requires_grad = False

    gate = ContextCEGate(width=24, max_corr=1.0).to(DEV)
    trainable = sum(p.numel() for p in gate.parameters() if p.requires_grad)

    tr = DataLoader(DS("train", True), batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    va = DataLoader(DS("val", False), batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    te = DataLoader(DS("test", False), batch_size=10, shuffle=False, num_workers=0, pin_memory=True)

    def softmass(x, t):
        return torch.sigmoid((x-t)/0.03).mean()

    def lossfn(logit, y, p, qbase):
        q = torch.sigmoid(logit)
        seg = F.binary_cross_entropy_with_logits(logit, y) + 0.8*soft_dice_loss(logit, y)
        mass = sum((softmass(q,t)-softmass(p,t)).abs() for t in (0.15,0.25,0.5))/3.0
        brier = ((q-y)**2).mean()
        anchor = (q-qbase).abs().mean()
        return seg + 0.50*mass + 0.12*brier + 0.18*anchor

    @torch.no_grad()
    def ev(loader):
        gate.eval()
        tp=fp=fn=0
        mae=0.0
        n=0
        masses={t:[0,0] for t in (0.15,0.25,0.5)}
        ctx_tp=ctx_fp=ctx_fn=0
        for rgb,p,y in loader:
            rgb,p,y = rgb.to(DEV),p.to(DEV),y.to(DEV)
            qb = context(rgb,p)["refined_prob"]
            c = torch.sigmoid(ce(rgb))
            r = torch.sigmoid(gate(p,qb,c,rgb))
            pr = r>=0.5
            gt = y>=0.5
            tp += int((pr&gt).sum())
            fp += int((pr&~gt).sum())
            fn += int((~pr&gt).sum())
            bp = qb>=0.5
            ctx_tp += int((bp&gt).sum())
            ctx_fp += int((bp&~gt).sum())
            ctx_fn += int((~bp&gt).sum())
            mae += (r-p).abs().sum().item()
            n += r.numel()
            for t in masses:
                masses[t][0] += int((r>t).sum())
                masses[t][1] += int((p>t).sum())
        i = tp/(tp+fp+fn+1e-9)
        f = 2*tp/(2*tp+fp+fn+1e-9)
        bi = ctx_tp/(ctx_tp+ctx_fp+ctx_fn+1e-9)
        bf = 2*ctx_tp/(2*ctx_tp+ctx_fp+ctx_fn+1e-9)
        sh = {str(t):(a-b)/max(1,b) for t,(a,b) in masses.items()}
        drift = float(np.mean([abs(v) for v in sh.values()]))
        sc = float(i - 0.08*drift)
        return {
            "iou": float(i),
            "f1": float(f),
            "context_iou": float(bi),
            "context_f1": float(bf),
            "prob_mae": float(mae/max(1,n)),
            "relative_mass_shift": sh,
            "selection": sc,
        }

    base = ev(va)
    best = base["selection"]
    torch.save({"epoch":0, "gate":gate.state_dict(), "val":base}, BEST)
    log(f"START DUTR-v7-ContextCEGate DEVICE={DEV} trainable={trainable}")
    log("baseline " + json.dumps(base))

    opt = torch.optim.AdamW(gate.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-5)

    with CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch","train_loss","val_iou","val_f1","context_iou",
            "prob_mae","shift15","shift25","shift50","selection"
        ])

    bad=0
    patience=8
    for ep in range(1,29):
        gate.train()
        sm=0.0
        sample_count=0
        for rgb,p,y in tr:
            rgb,p,y = rgb.to(DEV),p.to(DEV),y.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                qb = context(rgb,p)["refined_prob"]
                c = torch.sigmoid(ce(rgb))
            logit = gate(p,qb,c,rgb)
            loss = lossfn(logit,y,p,qb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
            opt.step()
            sm += loss.item()*rgb.size(0)
            sample_count += rgb.size(0)
        sched.step()
        v = ev(va)
        log(f"ep={ep:02d} loss={sm/sample_count:.5f} " + json.dumps(v))
        with CSV.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep, sm/sample_count, v["iou"], v["f1"], v["context_iou"], v["prob_mae"],
                v["relative_mass_shift"]["0.15"],
                v["relative_mass_shift"]["0.25"],
                v["relative_mass_shift"]["0.5"],
                v["selection"],
            ])
        if v["selection"] > best + 1e-5:
            best = v["selection"]
            bad = 0
            torch.save({"epoch":ep,"gate":gate.state_dict(),"val":v}, BEST)
            log(f"BEST epoch={ep} selection={best:.6f} iou={v['iou']:.6f}")
        else:
            bad += 1
            if bad >= patience:
                log(f"EARLY_STOP ep={ep}")
                break

    ck = torch.load(BEST, map_location=DEV, weights_only=False)
    gate.load_state_dict(ck["gate"])
    val = ev(va)
    test = ev(te)
    summary = {
        "best_epoch": int(ck["epoch"]),
        "val": val,
        "test_fixed": test,
        "trainable_params": trainable,
        "protocol": (
            "2D-only conservative gate around frozen DUTR-v5 Context2D; "
            "independent CE branch pretrained on CrackEnsembles train/calibration; "
            "gate trained on L515 train and selected on L515 val only; "
            "L515 test evaluated once after freeze; final CrackStructures GT unused; "
            "3D PAC/WFR/GTR/router untouched"
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("FINAL " + json.dumps(summary))

if __name__=="__main__": main()
