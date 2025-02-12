# 🎵 Projeto de Formatura: Processamento de Áudio com Redes Neurais

## 📌 Objetivo

Este projeto tem como objetivo desenvolver um software que permite editar, de forma separada, bateria, baixo, voz e outros instrumentos de uma música utilizando técnicas de processamento de áudio baseadas em redes neurais.

Nos últimos anos, avanços em inteligência artificial possibilitaram a separação de faixas de áudio em componentes individuais. No entanto, essas tecnologias ainda são pouco acessíveis para músicos e produtores. O objetivo deste projeto é criar um software intuitivo e eficiente para facilitar essa edição.

## 🛠️ Tecnologias Utilizadas

### 📌 Linguagem de Programação
- Python

### 📌 Bibliotecas
- `numpy` - Cálculo numérico
- `librosa` - Manipulação de arquivos de áudio
- `sounddevice` - Reprodução de áudio
- `customtkinter` - Interface gráfica

### 📌 Rede Neural
- **Demucs** - Separador de fontes de áudio baseado em arquitetura híbrida (forma de onda e espectrograma)

## 🖥️ Interface Gráfica

### Como Usar
1. Adicione um arquivo de áudio ao diretório do programa
2. Digite o nome do arquivo na caixa de texto
3. Clique no botão **"Separar"** para processar o áudio
4. Baixe a faixa editada clicando no botão **"Download"**, gerando o arquivo `audio_pronto.mp3`

## 📊 Modelos Utilizados

### **Hybrid Demucs**
- Criado pela Meta (2022)
- Separador de fontes baseado em redes neurais
- Separa um arquivo de áudio em quatro faixas: **voz, baixo, bateria e outros instrumentos**

### **Kernels LSTM**
- Redes neurais especializadas em aprender padrões sequenciais de longo prazo
- Melhoram a qualidade da separação de faixas

### **Transformer Encoder**
- Utiliza autoatenção para destacar partes relevantes da entrada
- Permite lidar com sequências de comprimento variável

## 📜 Arquitetura

O modelo **Hybrid Transformer Demucs** é uma evolução do Hybrid Demucs original. Ele combina:
- **Duas U-Nets** (operação no tempo e na frequência)
- **Transformer de atenção cruzada** para integrar os dados sonoros
- **Encoders e decoders** baseados em transformadores

Treinado com um dataset de **3500 músicas**, o modelo oferece uma separação precisa com alto **SDR (Signal to Distortion Ratio)**.

## 🔧 Estrutura do Código

- `signal.py` - Processamento do sinal de áudio
- `gui.py` - Interface gráfica

## 🎛️ Efeitos Implementados

### 🔊 Volume
- Multiplica o áudio por um coeficiente de volume ajustável (não linear)

### 🎚️ Filtro Butterworth
- Passa-baixa ou passa-alta, com frequência de corte e ordem definidas pelo usuário

### 🏛️ Reverberação
- Simula reflexões sonoras em uma sala grande

### 🔁 Eco
- Gera cópias repetidas do sinal com ajuste de tempo e decaimento

### 🎸 Distorção
- **Hardclip** e **Sigmoid**, ambas ajustáveis

### 🎵 Sintetizador
- Simula sons eletrônicos ou instrumentais com base em espectros de som

### 🎹 ADSR Envelope
- Controle dinâmico de **Ataque, Decaimento, Sustentação e Liberação** do som

### 🎶 Síntese Granular
- Manipula pequenos fragmentos de áudio para criar sons complexos

## 📚 Referências

- [Hybrid Transformers for Music Source Separation (arXiv)](https://arxiv.org/abs/2211.08553)

