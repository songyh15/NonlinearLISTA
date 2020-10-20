# Learning Fast Approximations of Sparse Nonlinear Regression
This repository has the code corresponding to the Paper 
"[Learning Fast Approximations of Sparse Nonlinear Regression]()".

The code is based on the LISTA-CPSS repository (https://github.com/xchen-tamu/linear-lista-cpss)
and is tested in Linux environment (Tensorflow: 1.10.0, CUDA9.0) with GTX 1080 GPU.

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```

```
## Testing
- Test NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero
```
python main.py -t -gpu 0 -id 23 -n 'NLISTA_cosx' -fun 'cosx' -S 'inf' -C 0.0 
```
- Test NLISTA in the case where the nonlinear function is 2x+cos(x) , signal-noise-radio(SNR) is 30dB and condition number is zero
```
python main.py -t -gpu 0 -id 23 -n 'NLISTA_cosx' -fun 'cosx' -S 30 -C 0.0 
```
- Test NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is 50
```
python main.py -t -gpu 0 -id 23 -n 'NLISTA_cosx' -fun 'cosx' -S 'inf' -C 50.0 
```
- For testing LISTA, replace 'NLISTA_cosx' with 'LISTA'.

- Test SpaRSA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero
```
python ISTA.py -model 'SpaRSA' -f '2xcosx' -mu 0.5 -SNR 'inf' -cond 0
```
- Test SpaRSA in the case where the nonlinear function is 2x+cos(x) , signal-noise-radio(SNR) is 30dB and condition number is zero
```
python ISTA.py -model 'SpaRSA' -f '2xcosx' -mu 0.5 -SNR 30 -cond 0
```
- Test SpaRSA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is 50
```
python ISTA.py -model 'SpaRSA' -f '2xcosx' -mu 0.5 -SNR 'inf' -cond 50
```
- For testing FISTA\FPCA\STELA, replace 'SpaRSA' with 'FISTA\FPCA\STELA'

## Training
- Train NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero
```
python main.py -gpu 0 -id 0  -n 'NLISTA_cosx' -fun 'cosx' -S 'inf' -C 0.0 
```
- Train NLISTA in the case where the nonlinear function is 2x+cos(x) , signal-noise-radio(SNR) is 30dB and condition number is zero
```
python main.py -gpu 0 -id 0  -n 'NLISTA_cosx' -fun 'cosx' -S 30 -C 0.0 
```
- Train NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is 50
```
python main.py -gpu 0 -id 0  -n 'NLISTA_cosx' -fun 'cosx' -S 'inf' -C 50.0 
```
- For training LISTA, replace 'NLISTA_cosx' with 'LISTA'.