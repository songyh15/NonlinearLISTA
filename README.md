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
python main.py -t -gpu 0 -id 0 -n 'NLISTA_2xcosx' -fun '2xcosx' -S 'inf' -C 0.0 
```
- For choosing other networks, set '-n' as 'LISTA\ NLISTA_10xcos2x\ NLISTA_10xcos3x\ NLISTA_10xcos4x'
- For choosing other nonlinear functions, set '-fun' as '10xcos2x\ 10xcos3x\ 10xcos4x'
- For choosing other noise levels, set '-S' as 30
- For choosing other condition numbers, set '-C' as 50.0


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
python main.py -gpu 0 -id 0  -n 'NLISTA_2xcosx' -fun 'cosx' -S 'inf' -C 0.0 
```
- For choosing other networks, set '-n' as 'LISTA\ NLISTA_10xcos2x\ NLISTA_10xcos3x\ NLISTA_10xcos4x'
- For choosing other nonlinear functions, set '-fun' as '10xcos2x\ 10xcos3x\ 10xcos4x'
- For choosing other noise levels, set '-S' as 30
- For choosing other condition numbers, set '-C' as 50.0