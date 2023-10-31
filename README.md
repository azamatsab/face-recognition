## Пайплайн для обучения FaceRecogniton 

###### Результаты обучения находятся в папке mlruns, чтобы с ходом обучения наберите следующую команду:
      mlflow ui

###### Файлы конфигурации находятся в папке confs и имеют иерархическую структуру

###### Помимо mlflow, каждый эксперимент логируется с помщью hydra, с логами можно ознакомиться в папке outputs
###### Веса моделей хранятся в outputs/<experiment_date>/weights <br />
<br />
<br />

#### График Train loss из mlruns
![alt text](https://github.com/azamatsab/face-recognition/blob/master/images/train_loss.png?raw=true)

<br />
<br />
<br />

#### График Val accuracy из mlruns
![alt text](https://github.com/azamatsab/face-recognition/blob/master/images/val_accuracy.png?raw=true)

<br />
<br />
<br />

#### График Val EER из mlruns
![alt text](https://github.com/azamatsab/face-recognition/blob/master/images/val_eer.png?raw=true)

<br />
<br />
<br />

#### График FAR/FRR из mlruns
![alt text](https://github.com/azamatsab/face-recognition/blob/master/images/far_frr.png?raw=true)

<br />
На графике выше, в связи с особенностями логирования в mlflow, пришлось умножить пороги на 10000, на самом деле пороги от 0 до 1

