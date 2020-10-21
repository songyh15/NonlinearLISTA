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
The test data is stored in the folder './data'.
The pretrained models are stored in the folder './experiments'.

Test NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero:
```
python main.py -t -gpu 0 -id 0 -n 'NLISTA_2xcosx' -fun '2xcosx' -S 'inf' -C 0.0 
```
- For choosing other networks, set '-n' as 'LISTA\ NLISTA_10xcos2x\ NLISTA_10xcos3x\ NLISTA_10xcos4x'
- For choosing other nonlinear functions, set '-fun' as '10xcos2x\ 10xcos3x\ 10xcos4x'
- For choosing other noise levels, set '-S' as 30
- For choosing other condition numbers, set '-C' as 50.0

Test SpaRSA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero:
```
python ISTA.py -model 'SpaRSA' -f '2xcosx' -mu 0.5 -SNR 'inf' -cond 0
```
- For choosing other algorithms, set '-model' as 'FISTA\ FPCA\ STELA'
- For choosing other nonlinear functions, set '-f' as '10xcos2x\ 10xcos3x\ 10xcos4x'
- For choosing other noise levels, set '-SNR' as 30
- For choosing other condition numbers, set '-cond' as 50
- For choosing other regularization parameters, modify '-mu'

## Training
Train NLISTA in the case where the nonlinear function is 2x+cos(x) , no noise exists and condition number is zero:
```
python main.py -gpu 0 -id 0  -n 'NLISTA_2xcosx' -fun '2xcosx' -S 'inf' -C 0.0 
```
- For choosing other networks, set '-n' as 'LISTA\ NLISTA_10xcos2x\ NLISTA_10xcos3x\ NLISTA_10xcos4x'
- For choosing other nonlinear functions, set '-fun' as '10xcos2x\ 10xcos3x\ 10xcos4x'
- For choosing other noise levels, set '-S' as 30
- For choosing other condition numbers, set '-C' as 50.0