# SEF-PNet

Official PyTorch implementation of the paper "[SEF-PNet: Speaker Encoder-Free Personalized Speech Enhancement with Local and Global Contexts Aggregation](https://arxiv.org/abs/2501.11274)" in ICASSP 2025.

## Веб Демо
[tse.ktsd.cc](https://tse.ktsd.cc/)


Можно взять примеры аудио в папке *audio/*

## В этом форке я пытаюсь сделать модель стриминговой

Для начала я решил сделать стриминговой модель [DPCCN](https://github.com/jyhan03/dpccn), которая является предшественником SEF-PNet. В ней нет части архитектурных решений из SEF-PNet.

С изначальной архитектурой модель не может работать в стриминге из-за неторых модулей:
* Куча свёрток дают большой лейтенси.
* Есть InstanceNorm, который нормализует по временной оси.
* Есть пулинг с интерплоляцией, который тоже даёт задержку. А интерполяцию не сделать в стриминге.


### Свёртки

Код тут [streaming_stuff.py](nnet/streaming_stuff.py)

Чтобы не ждать пока последняя свёртка накопит весь свой receptive field я сделал их каузальными, добавив паддинг размером `(kernel_size - 1) * dilation`. Так свёртка отрабатывает сразу, как получает первый фрейм и отдает выход в следующую свёртку, которая тоже отрабатывает сразу же. И того задержка в 1 фрейм получается.


Также я написал обёртку над свёртками, чтобы она хранила предыдущие фреймы в буфере и подавала их в свёртку.


## Нормализация

Код [streaming_norm.py](nnet/modules/streaming_norm.py)


Я попробовал несколько нормализаций:
* LayerNorm.
* InstanceNorm, который нормализует только по предыдущим кадрам.
* Тоже самое, что выше, но со скользящим средним для забывания (EMA).

Лучше всего из этого работало EMA, но все равно дает сильную просадку по метрикам. 


## Пулинг

Я сделал, чтобы среднее считалось не каждые pool_size кадров, а за последние pool_size кадров. 

Однако в оригинале там после каждого пулинга были свёртки (1x1). Этого я не добавил.

Но после замены я получаю эхо на аудио. Еще не разобрался с этим.

## Эквивалентность

Проверить эквивалентность стриминга и оффлайна можно скриптом

```sh
python nnet/compare_streaming_and_offline.py
```

## TODO
- [] Подумать над нормализацией.
- [] Пофиксить пулинг.
- [] Сконвертировать в onnx с прокидыванием буферов как входов.
- [] Померить перф на ноутбуке.
- [] Обучить в 48khz.
- [] Обучить на датасете DNS.

## Dataset
[Libri2Mix](https://github.com/JorisCos/LibriMix) min wav8k dataset. The `Data` folder contains three subfolders: `train`, `dev`, and `test`. Each subfolder includes three files:
- `mix_clean.scp`: Clean mixtures of 2 speakers.
- `ref.scp`: Target speaker’s speech.
- `auxs1.scp`: Enrollment speech from the target speaker, which is different from the target speaker’s speech in the mixture.

The `mix_clean.scp` corresponds to the **2-speaker** scenario in the results section.
Note that in this dataset, only the first speaker in the mixed speech is considered the target speaker.
Make sure to update the file paths in the `scp` files to match your local data locations. Also, remember to update the data paths in `conf_unet_tse_32ms.py` accordingly.

## Training
- **`train.sh`**: Shell script that initiates training by setting parameters (e.g., epochs, batch size, GPU settings) and calling the Python script (`train_unet_tse_steplr_clip.py`). To train the model, run:
  ```bash
  ./train.sh

- **`train_unet_tse_steplr_clip.py`**: Main Python script for training. It initializes the model, sets up data loaders, and manages the training loop.
   
- **`conf_unet_tse_32ms.py`**: Configuration file containing model architecture, data paths, and training hyperparameters.

- **`SEF_PNet_pse.py`**: Defines the `SEF_PNet` model, which is used in the training script.

## Evaluation

To evaluate the model, use the provided `eval.sh` script. It sets the necessary parameters (e.g., model checkpoint, GPU ID, data paths) and calls `evaluate.py` for performance evaluation.

- **`eval.sh`**: Runs the evaluation by setting paths and calling `evaluate.py`.  
  - Usage:
    ```bash
    ./eval.sh
    ```

- **`evaluate.py`**: Evaluates the model on the test set, computing metrics like SDR, SI-SNR, PESQ, and STOI.

## Results

Condition-wise results on three Libri2Mix PSE tasks:

<table>
  <thead>
    <tr>
      <th rowspan="2">Condition</th>
      <th rowspan="2">Method</th>
      <th colspan="3">Metrics</th>
    </tr>
    <tr>
      <th>SI-SDR</th>
      <th>PESQ</th>
      <th>STOI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" align="center"><strong>1-speaker+noise</strong></td>
      <td>Mixture</td>
      <td>3.27</td>
      <td>1.75</td>
      <td>79.51</td>
    </tr>
    <tr>
      <td>sDPCCN</td>
      <td>14.49</td>
      <td>3.04</td>
      <td>92.47</td>
    </tr>
    <tr>
      <td>SEF-PNet</td>
      <td>14.50</td>
      <td>3.05</td>
      <td>92.47</td>
    </tr>
    <tr>
      <td rowspan="3" align="center"><strong>2-speaker</strong></td>
      <td>Mixture</td>
      <td>-0.03</td>
      <td>1.60</td>
      <td>71.38</td>
    </tr>
    <tr>
      <td>sDPCCN</td>
      <td>11.62</td>
      <td>2.76</td>
      <td>87.19</td>
    </tr>
    <tr>
      <td>SEF-PNet</td>
      <td>13.00</td>
      <td>3.05</td>
      <td>89.71</td>
    </tr>
    <tr>
      <td rowspan="3" align="center"><strong>2-speaker+noise</strong></td>
      <td>Mixture</td>
      <td>-2.03</td>
      <td>1.43</td>
      <td>64.65</td>
    </tr>
    <tr>
      <td>sDPCCN</td>
      <td>6.93</td>
      <td>2.12</td>
      <td>79.32</td>
    </tr>
    <tr>
      <td>SEF-PNet</td>
      <td>7.54</td>
      <td>2.14</td>
      <td>80.58</td>
    </tr>
  </tbody>
</table>

### GPU Setup
This code is designed to run on a single GPU. By default, in the `train.sh` script, the `gpuid` is set to `0`. 

To use multiple GPUs, modify `gpuid=0,1,2,...` in `train.sh`. 

Additionally, for multi-GPU setups, comment out the line:
```python
from memonger import SublinearSequential
```
and replace SublinearSequential with nn.Sequential in SEF_PNet_pse.py to avoid memory issues.

### Create SCP
The SCP file I provided is from [DPCCN](https://github.com/jyhan03/icassp22-dataset/tree/main/lst/libri2mix). It only uses the first speaker as the target. To match MC-Spex results for the 2-speaker condition in Libri2Mix, you'll need to use double the data, with two speakers taking turns as the target. This means you’ll need to recreate the SCP files for training, validation, and testing. You can use the script in the link for reference.

Any problems, contact me at hzlkycg111@163.com, and a reply will be given promptly.
