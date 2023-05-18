# eFaceRecon
Reconhecedor e classificador facial em tempo real com OpenCV, dlib e Deep Learning

[![Python Version](https://img.shields.io/badge/python-3.7.8-green)](https://www.python.org/downloads/release/python-378/)
[![CMake Version](https://img.shields.io/badge/cmake-3.16.3-green)](https://cmake.org/cmake/help/v3.16/release/3.16.html)

~~Este programa é parte do Trabalho de Conclusão de Curso disponível para dowload em: http://repositorio.ufla.br/jspui/handle/1/44811~~

### Execução

1. Entre na pasta do projeto e crie um virtualenv com o python
```
$ cd eFaceRecon
$ virtualenv -p python3 env
```

2. Entre no virtualenv
```
$ source env/bin/activate
```

3. Instale as dependências do pip:
```
$ pip install -r requirements.txt
```

4. Baixe o arquivo <https://drive.google.com/file/d/1dQSw-CA-BMtWWgp-EcXlA3_ZE_gw7LDa/view?usp=sharing>, grave-o na pasta do projeto e execute os comandos abaixo:

```
$ tar -zxvf arquivos_grandes.tar.gz
$ rm arquivos_grandes.tar.gz
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
### Configuração
O arquivo de configuração `config.ini` é utilizado para definir a combinação de métodos que compõem a arquitetura geral do sistema. Leia os comentários do arquivo para saber como configurá-lo corretamente.


### Arquivos de Teste

Teste disponível no arquivo `test.ipynb`. Para acessá-lo, execute o comando abaixo e, na página que abrir no seu navegador, clique no arquivo para abrí-lo. Para mais informações, consulte: <https://jupyter-notebook.readthedocs.io/en/stable/> 
```
$ jupyter notebook
```
OBSERVAÇÃO: O programa somente executa corretamente com uma ou mais pessoas registradas na base.
