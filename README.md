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
4. Instale as dependências do pip:
```
$ pip install -r requirements.txt
```

5. Para tirar amostras de uma face:
```
$ python face_gatherer.py
```

6. Para treinar o modelo novamente:
```
$ python trainer.py
```

7. Para testar a detecção de faces:
```
$ python efacerecon.py
```

8. Teste disponível no arquivo `test.ipynb`. Para acessá-lo, execute o comando abaixo e, na página que abrir no seu navegador, clique no arquivo para abrí-lo. Para mais informações, consulte: <https://jupyter-notebook.readthedocs.io/en/stable/> 
```
$ jupyter notebook
```

Obs: pasta models não disponível. Para baixar o arquivo landmarks.dat entre em <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>.
