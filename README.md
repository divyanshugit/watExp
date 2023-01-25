# watExp
### Create Environment

```bash
conda create -n <environment name> python==3.7.0
pip install -r requirements.txt
```

### Data Generation

```py
$ python generate_parallelData.py --f_lang Hindi --s_lang <Indic Language>
```
### Training:

```py
$ python train.py --s_lang <Indic Language> --t_lang <Indic Language> --gpu <GPU Number>
```
