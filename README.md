# eFaceRecon
Reconhecedor e classificador facial em tempo real com OpenCV, dlib e Deep Learning

**versão do python**: 3.7 e 3.7-dev

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

5. Baixe o arquivo <https://drive.google.com/file/d/1dQSw-CA-BMtWWgp-EcXlA3_ZE_gw7LDa/view?usp=sharing>, grave-o na pasta do projeto e execute os comandos abaixo:

```
$ tar -zxvf arquivos_grandes.tar.gz
$ rm arquivos_grandes.tar.gz
```

6. Para tirar amostras de uma face:
```
$ python face_gatherer.py
```

7. Para treinar o modelo novamente:
```
$ python trainer.py
```

8. Para testar a detecção de faces:
```
$ python efacerecon.py <rede_neural> <classificador>
Exemplo:
$ python efacerecn.py nn4 knn
```

9. Teste disponível no arquivo `test.ipynb`. Para acessá-lo, execute o comando abaixo e, na página que abrir no seu navegador, clique no arquivo para abrí-lo. Para mais informações, consulte: <https://jupyter-notebook.readthedocs.io/en/stable/> 
```
$ jupyter notebook
```

