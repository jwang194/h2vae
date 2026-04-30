# Training-loop speedups for h2vae

Each item lists: what the change is, where it lives, expected magnitude, branches affected, VRAM trade-off, and numerical-accuracy trade-off. "Branches" refers to model variant (`vae1d`/`vae2d`/`vae3d`), heritability mode (`--kinship` / genotype / `--genetic-correlation`), `--split-variants`, and streaming vs. in-memory data.

---

## Tier 1 — Large, low-risk wins

### 1. Vectorize per-dimension display heritability
**Where:** `train_hvae.py:567-580` (`_compute_her_estimates`).
**Today:** `[her_fn(Z[:, i:i+1]).item() for i in range(zdim)]` — `zdim` (e.g. 64) sequential calls, each materializing per-dim quad forms / projections.
**Change:** call `her_fn(Z)` once — the `mom`, `var_exp`, and `gc` callables in `h2vae/heritability.py` are *already batched* (return a `(zdim,)` vector). The per-dim loop is redundant.
**Speedup:** ~zdim-fold reduction in per-epoch display cost. Display is run twice on train (even + odd) and twice on val when `--split-variants`, so the savings compound. The biggest win is in `--kinship` and `gc` modes where each call does `(n,n) @ (n,1)` mvms — turning 64 mvms into one `(n,n)@(n,d)` GEMM.
**Branches:** all (kinship + genotype + gc, with/without split-variants).
**VRAM:** essentially unchanged (Z is already (n, zdim) on device).
**Numerical accuracy:** identical results — same algebra, same dtype.

### 2. Stop re-encoding the train split for display every epoch
**Where:** `train_hvae.py:683-685` ("Re-encode with current weights ...").
**Today:** every epoch ends with a *third* full encode pass (`encode_all` on `train_loader`) just to produce posterior means for stable h² display and the per-epoch `Zm_train.NNNNN.txt` dump.
**Change:** either (a) only do the post-train re-encode + dump every `cfg.epoch_cb` epochs (matches checkpoint cadence; the curves in `out/plot_heritability.py` only need points where checkpoints exist), or (b) reuse the `Zm` already collected at the *start* of the next epoch (it sees the freshly-updated weights). Option (a) is the simplest.
**Speedup:** removes ~33% of forward-pass time per epoch (3 encode passes → 2). For `vae3d` on UKB MRI this is a substantial wall-clock win.
**Branches:** all.
**VRAM:** unchanged.
**Numerical accuracy:** display-only; per-epoch h² values now appear on a coarser grid. No effect on optimization.

### 3. Save latents as `.npy`, not tab-delimited text
**Where:** `train_hvae.py:687-691, 720-724` and `eval_latents.py:151-152`.
**Today:** `np.savetxt(..., delimiter="\t")` formats every float to ASCII every epoch. For `n=20k`, `zdim=64` that's ~1.3M format calls per dump, twice per epoch (train + val). Often a real fraction of epoch time on shared filesystems.
**Change:** `np.save` (or `np.savez`) — binary, ~10–100× faster, ~3× smaller files. Update `eval/helpers.py` consumers (anything that does `np.loadtxt(Zm_*.txt)`) to use `np.load`.
**Speedup:** seconds → milliseconds per dump. Bigger if dumping every epoch is kept; combine with #2 for largest effect.
**Branches:** all. Also affects the `eval/` Snakemake pipeline downstream readers — coordinated change.
**VRAM:** unchanged.
**Numerical accuracy:** improves slightly (binary preserves full float32; `savetxt` default `%.18e` is full but `loadtxt` parsing can introduce tiny rounding).

### 4. Enable cuDNN benchmark + TF32
**Where:** add at top of `train_hvae.py` `main()`.
**Change:**
```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
```
**Speedup:** 10–30% on the conv path (cuDNN benchmark picks the best algo for the fixed input shape). TF32 helps the dense projection (`dense_zm`/`dense_zs`/`dense_dec`) and any large heritability matmul.
**Branches:** all (vae3d benefits most because conv volume is huge).
**VRAM:** unchanged or slightly higher (cuDNN may pick algorithms with bigger workspaces).
**Numerical accuracy:** TF32 truncates matmul mantissas to 10 bits — minor loss on convs/dense. Heritability computations on (n,n) matrices live in standalone GEMMs and *will* lose a little precision; if the trace/quadratic-form values matter at >3 decimals you can keep float32 there with `with torch.backends.cuda.matmul.allow_tf32 = False:` around `setup_heritability` and the loss callables. cuDNN benchmark itself is bit-exact for fixed shapes.

### 5. Fold validation MSE into the existing val encode loop
**Where:** `train_hvae.py:704-746`. `validate_epoch` first calls `encode_all(val_loader)`, then iterates `val_loader` *again* to compute MSE.
**Change:** in a single pass per val batch, run `encode` + `decode` (use `zm` directly, no sampling needed for display) + accumulate MSE. Streaming pipelines (NIfTI especially) pay the file-IO cost once instead of twice.
**Speedup:** halves val IO and forward-pass time. For NIfTI/streaming branches this is meaningful — gzip decompression dominates val time.
**Branches:** all; magnitude largest with NIfTI streaming.
**VRAM:** unchanged.
**Numerical accuracy:** identical.

### 6. Mixed precision (AMP / bfloat16)
**Where:** training step in `train_hvae.py:647-678` and the encode passes.
**Change:** wrap forward in `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` (or `fp16` with `GradScaler`). bfloat16 is preferred on Ampere/Hopper — same exponent range as float32, no scaler needed.
**Speedup:** typically 1.5–2.5× wall-clock for conv-heavy nets, and ~halves activation memory (lets you bump `--bs`). vae3d is the prime beneficiary.
**Branches:** all model branches. Heritability loss matmuls (`mom`, `var_exp`, `gc`) should remain in float32 — wrap *only* the encode/decode in autocast and cast `zm`, `zs`, `Z` back to float32 before the heritability call. The closures in `heritability.py` do quadratic forms on (n,n) tensors where fp16 underflow is plausible.
**VRAM:** ~2× reduction on activations.
**Numerical accuracy:** bfloat16 has ~7-bit mantissa — small per-step noise; usually washes out in training. Keeping heritability + KL term in float32 protects the regularization objective. Watch for occasional NaNs on the softplus-of-`dense_zs` at fp16; bfloat16 avoids this.

---

## Tier 2 — Medium wins

### 7. Bigger batch size for encode-only passes
**Where:** `encode_all` in `train_hvae.py:587-612`.
**Today:** uses `train_loader` / `val_loader` with the same `cfg.bs` used for backprop. Encode-only has no activation graph and uses `no_grad`, so VRAM headroom is much larger.
**Change:** create a second `DataLoader` with `batch_size=cfg.bs * 4` (or similar) for encode passes. Or inside `encode_all` re-batch by indexing the whole tensor in chunks.
**Speedup:** kernel-launch and host overhead go down; bigger conv workloads are more compute-bound. Often 1.2–1.5× on the encode portion.
**Branches:** all; biggest for vae3d where launch overhead per small batch is high.
**VRAM:** rises during encode (still fits because no backward graph).
**Numerical accuracy:** identical.

### 8. Use `non_blocking=True` host→device copies
**Where:** every `data[0].to(device)` (encode_all, train_epoch, validate_epoch).
**Change:** `data[0].to(device, non_blocking=True)` — `pin_memory=True` is already on, so this enables true async copy overlapping the previous batch's compute.
**Speedup:** small but free; useful when input I/O is non-trivial (NIfTI/streaming).
**Branches:** all (most useful for streaming).
**VRAM:** unchanged.
**Numerical accuracy:** identical.

### 9. Avoid the `Z.detach().clone()` allocation each minibatch
**Where:** `train_hvae.py:478-482` in `compute_heritability_loss`.
**Today:** every minibatch allocates a fresh `(n_train, zdim)` clone purely as a defensive copy.
**Change:** since the caller already does `Z = Z.detach()` then `Z[idxs] = z` (line 648, 667), Z's storage is already detached except for the live batch slice. The clone in `compute_heritability_loss` is double work. Drop it; pass `Z` directly. (Validate that backward through the in-place index assignment doesn't complain — it shouldn't, since the source Z came from `no_grad` `encode_all`.)
**Speedup:** small per-minibatch saving; relevant when n_train is large (hundreds of MB allocated/freed per minibatch on big cohorts × many minibatches per epoch).
**Branches:** all where `--h-weight > 0`.
**VRAM:** small reduction.
**Numerical accuracy:** identical.

### 10. Cache PCA/inverse projections in `setup_heritability` paths that depend on covariates
**Where:** `mom(... C=...)` in `heritability.py:86-111`.
**Today:** materializes `P = I - C(C'C)^{-1}C'` as a dense (n, n), and `PKP`, `PKP @ PKP` — three (n, n) GEMMs at setup. Fine — but inside the loss closure, `(P @ ypp)` computes a full (n,n)@(n,d) every call; never materialize P.
**Change:** apply P implicitly: `P_apply(x) = x - C @ (CTCI @ (C.T @ x))`. Then both `(PKP @ ypp)` and `(P @ ypp)` are (c,c)/(c,d) work instead of (n,n)/(n,d). Already done this way in `var_exp` and `gc`; just propagate the pattern to the covariate branch of `mom`.
**Speedup:** big on large kinship cohorts when `--residualize-covariates` is on. Roughly `O(n*c*d)` instead of `O(n^2*d)`.
**Branches:** `--kinship` + `--residualize-covariates`. (Default kinship UKB run uses this.)
**VRAM:** drops — no longer keep P, PKP2 around (still need K and PKP for trace, but those are setup-once).
**Numerical accuracy:** same algebra in different order — small floating-point noise at most.

### 11. Replace `torch.linalg.solve` for the 2x2 system with closed-form
**Where:** `heritability.py:108`.
**Today:** `torch.linalg.solve(A, B)` for a fixed 2x2 A, called per minibatch.
**Change:** A is known at setup; precompute `A_inv` once and use `V = A_inv @ B`. Or write the 2x2 inverse explicitly.
**Speedup:** trivial per call but called many times; mostly removes the `linalg.solve` kernel-launch overhead.
**Branches:** `--kinship` + `--residualize-covariates`.
**VRAM:** unchanged.
**Numerical accuracy:** essentially identical for 2x2.

### 12. Replace `torch.trace(K @ K)` with `(K * K).sum()`
**Where:** `heritability.py:77` (and analogous spots in the `gc` setup if any).
**Today:** materializes K@K (n×n) just to take its trace; setup-only but n can be 50k+ in UKB.
**Change:** for symmetric K, `tr(K K) = sum(K^2)`. Saves an O(n^3) GEMM and an n×n allocation.
**Speedup:** big *one-time* setup speedup on large cohorts; not per-epoch.
**Branches:** `--kinship` (no covariates path).
**VRAM:** removes a transient (n,n) tensor at setup.
**Numerical accuracy:** sum-of-squares is more numerically stable than computing the full product and then tracing.

### 13. `torch.compile` the model
**Where:** after `vae = ModelClass(...).to(device)`, add `vae = torch.compile(vae)`.
**Speedup:** typically 10–30% on conv VAEs; sometimes more on vae3d. First epoch pays a compilation cost.
**Branches:** all.
**VRAM:** small overhead; sometimes lower due to operator fusion.
**Numerical accuracy:** identical (default `mode="default"` keeps fp32 ops). Combine with TF32/AMP for stacking gains.

### 14. Precompute and re-use streaming NIfTI cohort
**Where:** `NiftiFileDataset.__getitem__` in `h2vae/data.py:112-134`.
**Today:** every epoch re-decompresses each `.nii.gz` / `.nii.zst` via nibabel + zstd, z-scores, pads. With ~30k subjects this is the dominant epoch cost.
**Change (offline):** preprocess once into a single HDF5 (or shard of HDF5s) with pre-padded, pre-z-scored float32 volumes; load via the in-memory `ImageDataset` path or memory-mapped reads. If memory is tight, write each sample as a separate `.npy` (no decompression).
**Change (in-loop, smaller win):** `np.asarray(nib.load(p).dataobj, dtype=np.float32)` is faster than `.get_fdata()` since it skips caching machinery.
**Speedup:** offline preprocessing typically removes 50–80% of NIfTI-branch epoch time.
**Branches:** vae3d / NIfTI streaming.
**VRAM:** unchanged (still streamed); host RAM/disk grows for the preprocessed cache.
**Numerical accuracy:** identical (preprocessing reproduces the per-call math).

### 15. `optimizer.zero_grad(set_to_none=True)` (default in modern PyTorch)
**Where:** `train_hvae.py:674`.
**Change:** explicitly pass `set_to_none=True` (or rely on default in PyTorch ≥2.0).
**Speedup:** marginal — skips a kernel-launch zero-fill on each grad tensor.
**Branches:** all.
**VRAM:** unchanged.
**Numerical accuracy:** identical.

---

## Tier 3 — Smaller / situational wins

### 16. Hoist KL recomputation into the model
**Where:** `train_hvae.py:661-664` reimplements KLD inline; the model already has identical code in `forward()`. The training loop calls `encode` + `decode` separately rather than `forward`. If you call `forward`, the KL is computed once with no Python overhead.
**Speedup:** trivial — but eliminates duplication.
**Branches:** all.
**VRAM/accuracy:** unchanged.

### 17. Move in-memory image tensor to GPU once
**Where:** `ImageDataset` returns CPU tensors; per minibatch `data[0].to(device)` triggers an H→D copy. For non-streaming runs where `train_images` and `val_images` fit in GPU memory, push them once at startup.
**Change:** load `train_dataset.images` to `device` after construction; have `__getitem__` return a GPU tensor.
**Speedup:** cuts H→D PCIe traffic per minibatch. Real impact on small-image / large-batch runs (e.g. T1 2D slices at 256×256).
**Branches:** non-streaming + 2D / 1D models.
**VRAM:** **higher** — adds `n × c × h × w × 4` bytes on GPU. Often the budget allows it; check before flipping.
**Numerical accuracy:** identical.

### 18. Avoid building dataset Python tuples per item; batch index instead
**Where:** `ImageDataset.__getitem__`. PyTorch's collate loops over indices in workers. For in-memory mode you can use `torch.utils.data.TensorDataset`-style batched indexing in the main process and skip workers entirely.
**Speedup:** small — removes worker IPC overhead. Worth flipping `num_workers=0` for the in-memory branch and benchmarking; with images already on CPU/GPU there's no IO to hide.
**Branches:** non-streaming.
**VRAM:** unchanged.
**Numerical accuracy:** identical.

### 19. `num_workers` / `prefetch_factor` are hardcoded
**Where:** `train_hvae.py:858-864`. `nw=8` for image, `nw=16` for nifti.
**Change:** expose as a CLI flag; benchmark on the actual cluster node. On dual-socket hosts with many cores you may want 24+ for nifti; on small VMs, 4 may be ideal.
**Speedup:** tuned correctly, can shave 10–30% off streaming epochs.
**Branches:** streaming (especially NIfTI).
**VRAM:** unchanged. Host RAM grows with `prefetch_factor * num_workers * batch_size`.
**Numerical accuracy:** identical.

### 20. Skip the per-epoch latent dump entirely except at checkpoint epochs
**Where:** `train_hvae.py:687-691, 720-724`.
**Change:** wrap both `np.savetxt` calls in `if epoch % cfg.epoch_cb == 0:`. The `eval/` pipeline only consumes checkpoint epochs, and `out/plot_heritability.py` parses log lines, not these files.
**Speedup:** removes per-epoch text I/O entirely except every 10 epochs.
**Branches:** all.
**VRAM:** unchanged.
**Numerical accuracy:** unchanged; loses fine-grained latent snapshots that nothing currently consumes.

### 21. Cache `Eps` per-batch only
**Where:** `train_hvae.py:637-638` allocates `Eps = torch.randn_like(Zs)` of shape (n_train, zdim) on GPU each epoch.
**Change:** sample `eps` per minibatch (existing `eps = Eps[idxs]` pattern can be replaced with `eps = torch.randn_like(zs)` inside the loop).
**Speedup:** small — saves an n_train×zdim allocation. The sole reason to keep epoch-level Eps is reproducibility of the encode-phase Z used in `compute_heritability_loss(Z, ...)`. If posterior means (Zm) are used as the "frozen" reference (with eps only for the live batch's z), the per-epoch `Eps` becomes unnecessary. Worth a small numerical sanity-check to confirm the heritability targets aren't sensitive to noise on the *frozen* slice.
**Branches:** all.
**VRAM:** small reduction.
**Numerical accuracy:** changes the noise applied to non-batch latents within the heritability term — small change to the gradient through Z. May or may not matter; benchmark.

### 22. Async checkpoint write
**Where:** `train_hvae.py:921-923`.
**Change:** offload `torch.save` to a background thread (`concurrent.futures.ThreadPoolExecutor`), or save state_dict to CPU first asynchronously.
**Speedup:** tiny — only matters when storage is slow. On Hoffman2 with `/u/scratch` this is real.
**Branches:** all.
**VRAM:** unchanged; doubles host RAM during the copy.
**Numerical accuracy:** identical.

---

## Cross-cutting notes

- **Stacking:** items 1, 2, 3, 4, 5 stack additively and are essentially free (no accuracy/VRAM cost worth worrying about). Item 6 (AMP) and item 13 (`torch.compile`) stack on top, and together typically deliver another 1.5–2×.
- **Heritability remains the precision-sensitive part.** When pursuing AMP/TF32, keep `mom`, `var_exp`, `gc` callables in float32 — they perform large quadratic forms whose relative scale is what the loss reads, and downcasting can flip signs near zero. Cast `Z`/`zm` back to float32 at the boundary.
- **Item 14 (preprocess NIfTI offline)** is the single biggest lever for the vae3d / UKB MRI branch and is independent of the loss-side optimizations.
- **Validate per-branch:** the `--genetic-correlation` (`gc`) path was added more recently and uses `K @ V_y1` per call; items 1, 4, 6 all benefit it equally, but check that the closed-form clamp at `heritability.py:349` is exercised under bf16 (no underflow surprises).
- **`VarExpTaylorFactory`** in `heritability.py:208` is documented in CLAUDE.md but not invoked by `train_hvae.py:setup_heritability` (which uses full `var_exp`). If/when Taylor mode is wired in, it already does the right precomputation pattern — no change needed.
