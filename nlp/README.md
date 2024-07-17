# NLP

Aulas de processamento de linguagem natural.

## Setup information

Para rodar esse projeto, siga os passos abaixo.

### Iniciar o container do projeto

Para iniciar o container do projeto você pode usar o DOCKER CLI.

#### Docker CLI

Você precisa primeiramente contruir a imagem do projeto:

```bash
$ sudo docker build --tag=nlp:1.0 .
```

Depois você pode iniciar o container com os seguintes comandos:

```bash
$ sudo docker create -t -i --name nlp \
    -p 8000:8000 -v $PWD:/home/user \
    nlp:1.0

$ sudo docker start nlp
$ sudo docker exec -it nlp bash
$ jupyter lab --ip 0.0.0.0 --port 8000 --allow-root
```

#### Dev Container

Você também pode usar o dev container se preferir, basta iniciá-lo.