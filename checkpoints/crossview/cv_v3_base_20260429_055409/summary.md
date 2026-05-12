# Run Summary — cv_v3_base_20260429_055409

## Config
- **Backbone:** `convnext_base.fb_in22k_ft_in1k_384`
- **Embed dim:** 0
- **Image size:** 384
- **Batch size:** 16
- **Epochs:** 36
- **LR:** 0.0001 (weight_decay=0.0001)
- **Temperature:** init=0.07, learnable=True
- **Label smoothing:** 0.1
- **Hard negatives:** enabled=True, gps_epochs=4, pool_size=32, dss_refresh_every=1
- **Eval directions:** ['g2s', 's2g']
- **Eval metrics:** recall_ks=[1, 5, 10], map_k=1000

## Runtime
- Total wall time: **990.5 min** (59429 s)

## Best Epoch
- Epoch: **36**  (selected on `g2s_recall@1`)
- Score: **0.0858**
- **G2S**: recall@1=0.0858, recall@5=0.1813, recall@10=0.2229, map@1000=0.1325
- **S2G**: recall@1=0.0540, recall@5=0.1110, recall@10=0.1370, map@1000=0.0185

## Final Epoch
- Epoch: **36**
- **G2S**: recall@1=0.0858, recall@5=0.1813, recall@10=0.2229, map@1000=0.1325
- **S2G**: recall@1=0.0540, recall@5=0.1110, recall@10=0.1370, map@1000=0.0185

## Files
- `config.yaml` — resolved config used for this run
- `train_curves.csv` — per-epoch training stats
- `eval_curves.csv` — per-epoch evaluation metrics
- `best.pt`, `last.pt` — checkpoints
- `best_metrics.json`, `final_metrics.json` — metrics snapshots
