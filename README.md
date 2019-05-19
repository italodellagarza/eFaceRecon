# eFaceRecon
Reconhecedor e classificador facial em tempo real com OpenCV, dlib e Deep Learning

### Execução

1. Instale o cmake:

<https://cmake.org/>

2. Entre na pasta do projeto e crie um virtualenv com o python
```
$ cd eFaceRecon
$ virtualenv -p python3 env
```

3. Entre no virtualenv
```
$ source env/bin/activate
```
Instale as dependências do pip:
```
$ pip install -r requirements.txt
```

Obs: pasta models não disponível. Para baixar o arquivo landmarks.dat entre em <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>.