# GeJay — guião de aula

- **Ideia central:** transformar movimentos das mãos num instrumento musical, usando uma câmara e Python.
- **Objetivo da aula:** compreender como ligar visão por computador, geometria, interação e síntese de áudio num sistema em tempo real.
- **Pré-requisitos úteis:** funções, classes, listas, dicionários e noções básicas de coordenadas e ondas sonoras.

## 1. Apresentar o projeto

- O GeJay deteta até duas mãos, acompanha pontos dos dedos e calcula quatro distâncias.
- Essas distâncias tornam-se controlos musicais: altura da nota, brilho, reverberação e volume.
- O instrumento é monofónico: produz uma nota de cada vez; os dois canais de saída recebem o mesmo som.
- O projeto permite visualizar o seguimento, transmitir medidas por TCP e gravar vídeo com o áudio do sintetizador.
- Pergunta de abertura: «Como podemos tocar música sem tocar fisicamente num instrumento?»

## 2. Explicar o percurso dos dados

- **Câmara → imagem → MediaPipe → pontos das mãos → distâncias → valores entre 0 e 1 → parâmetros musicais → áudio.**
- A câmara fornece imagens; o modelo identifica pontos de referência, chamados *landmarks*.
- O código calcula características geométricas e associa-as a parâmetros do sintetizador.
- A inteligência artificial está no modelo de deteção e seguimento já treinado; o projeto não treina um modelo novo.
- A associação entre distâncias e som é definida por regras e fórmulas no código.

## 3. Bibliotecas e ferramentas importantes

- **OpenCV (`cv2`; `opencv-python`):** abrir a câmara, ler imagens, converter cores e desenhar a interface. Aprender que o OpenCV usa BGR e que o MediaPipe recebe RGB.
- **MediaPipe (`mediapipe`):** detetar mãos e estimar 21 pontos por mão. Aprender a consultar os pontos e a identificação de mão esquerda/direita.
- **NumPy (`numpy`):** representar imagens e sinais como arrays e calcular blocos de áudio. Aprender dimensões, tipos numéricos, operações vetorizadas e *broadcasting*.
- **sounddevice:** enviar os blocos de áudio para o dispositivo de saída através de um *callback*, usando PortAudio. Aprender taxa de amostragem, tamanho do bloco e continuidade da reprodução.
- **Biblioteca padrão de Python:** `math` para fórmulas; `dataclasses` para parâmetros; `threading` e `queue` para concorrência; `time` para medição; `argparse` para opções de execução; `socket` e `json` para comunicação; `pathlib` e `subprocess` para ficheiros e processos.
- **FFmpeg:** ferramenta externa usada para codificar a gravação MP4 com vídeo H.264 e áudio AAC; não é uma biblioteca Python.
- **`unittest`:** testes automáticos que permitem verificar partes do sistema sem câmara nem dispositivo de áudio.
- **Ambientes virtuais e dependências:** compreender `requirements.txt` e `requirements-synth.txt`. As versões fixadas da parte de visão foram preparadas para Python 3.10/3.11; a instalação deve seguir as dependências do projeto.

## 4. Visão por computador e geometria

- Cada mão tem 21 pontos; os pontos **4** e **8** correspondem às pontas do polegar e do indicador.
- O projeto usa as coordenadas **2D normalizadas `(x, y)`** desses pontos para calcular distâncias.
- Distância euclidiana: `d = sqrt((x2 - x1)^2 + (y2 - y1)^2)`.
- As quatro medidas são: indicador–indicador, polegar–polegar, indicador–polegar esquerdo e indicador–polegar direito.
- Normalização dos controlos: `u = (d - mínimo) / (máximo - mínimo)`, limitando o resultado ao intervalo `[0, 1]`.
- Exemplo: com mínimo `0,01`, máximo `0,25` e distância `0,13`, o controlo normalizado é `0,5`.
- Distinguir coordenadas normalizadas da imagem e normalização das distâncias para controlo: são duas operações diferentes.
- As medidas não são centímetros nem distâncias físicas 3D; variam com o enquadramento, a perspetiva e a posição relativamente à câmara. A escala diferente dos eixos da imagem também influencia a métrica.
- Calibrar os mínimos e máximos em `config/distances_config.py` para os movimentos confortáveis do utilizador.
- Distinguir **zero** de **ausência de deteção**: uma pinça fechada é uma medida válida; `None`/`null` significa que falta informação.

## 5. Dos gestos à música

- **Afastar os indicadores:** aumenta a altura da nota; por omissão, usa uma escala pentatónica de Dó maior entre C3 e C5.
- **Afastar os polegares:** aumenta o brilho, elevando o parâmetro de corte espectral entre 200 e 8000 Hz.
- **Abrir a pinça esquerda:** aumenta a reverberação, até 55% de mistura com o efeito.
- **Abrir a pinça direita:** aumenta o volume; abaixo do limiar normalizado de `0,08`, a voz deixa de ser excitada e a reverberação pode continuar brevemente.
- As medidas são tratadas separadamente no código, mas os movimentos físicos podem alterar várias ao mesmo tempo.
- Comparar controlo contínuo de altura com seleção de notas de uma escala.
- Explicar **histerese**: exigir uma margem extra antes de mudar de nota evita alternâncias causadas por pequenas oscilações do seguimento.
- Explicar **suavização**: transições graduais dos parâmetros reduzem mudanças bruscas e cliques; existe um compromisso entre estabilidade e resposta rápida.
- Distinguir número de nota MIDI de envio de mensagens MIDI: aqui o número é usado para calcular a frequência, `f = 440 × 2^((nota - 69) / 12)`.

## 6. Como se produz o som

- O sintetizador soma componentes sinusoidais harmónicas: é uma forma de **síntese aditiva**.
- A componente fundamental determina a altura; os harmónicos contribuem para o timbre.
- O brilho é controlado pela atenuação dos harmónicos através de uma envolvente espectral passa-baixo.
- A reverberação usa linhas de atraso com realimentação e filtros *all-pass*.
- A configuração inicial usa **48 000 amostras por segundo** e **blocos de 256 amostras**.
- Um bloco de 256 amostras a 48 kHz corresponde a cerca de **5,3 ms de áudio**; isto não é a latência total do sistema.
- Introduzir frequência de Nyquist e *aliasing*: o motor suprime harmónicos próximos do limite de amostragem.

## 7. Tempo real, desempenho e robustez

- A captura da câmara corre numa thread e guarda apenas a imagem mais recente, evitando acumular gestos antigos.
- O áudio é produzido independentemente da inferência da imagem; o *callback* não deve esperar por rede, escrita em disco ou processamento da câmara.
- A gravação usa filas com tamanho limitado e escrita/codificação em segundo plano.
- **FPS e latência são diferentes:** FPS mede frequência de atualização; latência mede o atraso entre um gesto e a resposta sonora.
- A 30 FPS, há cerca de 33 ms entre imagens; um gesto muito rápido pode acontecer entre duas capturas.
- Pedir 60 FPS não garante obter 60 FPS: é necessário medir e verificar o modo suportado pela câmara.
- O modo `--profile` mede desempenho interno, mas não mede toda a latência entre movimento e som.
- Ao perder uma mão, ou ficar 300 ms sem atualização de controlo, o sistema desvanece a saída, incluindo a reverberação.
- Discutir o compromisso entre resolução, precisão do modelo, iluminação, FPS e atraso.

## 8. Orientar a leitura do código

- `scripts/hand_distance_tracker.py`: ponto de entrada e ligação entre os componentes.
- `utils/hand_tracker.py`: utilização do MediaPipe e extração dos pontos.
- `utils/distance_calculator.py`: geometria e normalização.
- `synth/controls.py`: associação entre distâncias e parâmetros musicais.
- `synth/engine.py`: geração e reprodução de áudio.
- `tracking/camera.py` e `tracking/performance.py`: captura concorrente e medição de desempenho.
- `recording/`: interface de gravação, filas e sincronização de áudio/vídeo.
- `config/`: parâmetros ajustáveis; `tests/`: verificação automática do comportamento.
- Destacar separação de responsabilidades: tornar a matemática e o mapeamento testáveis sem depender do equipamento.

## 9. Demonstração sugerida

- Executar `python scripts/hand_distance_tracker.py --synth`.
- Mostrar as duas mãos e abrir a pinça direita para começar a ouvir som; demonstrar cada controlo com movimentos lentos.
- Executar `python scripts/hand_distance_tracker.py --synth --scale continuous` para comparar com a escala pentatónica.
- Retirar uma mão da imagem e observar o desaparecimento gradual do som.
- Usar `--profile` para relacionar a experiência com as medições de desempenho.
- Premir `r` para iniciar/parar uma gravação e `q` para sair; aguardar a mensagem de gravação concluída.
- Se não houver câmara ou saída de áudio disponível, gerar uma demonstração offline: `python scripts/local_synth.py --render-demo /tmp/gejay-demo.wav`.
- Mostrar a verificação automática: `python -m unittest discover -s tests -v`.

## 10. Exercícios e resultados de aprendizagem

- **Geometria:** calcular uma distância e normalizá-la à mão; confirmar o resultado no código.
- **Calibração:** ajustar os limites a dois utilizadores e explicar por que motivo a experiência muda.
- **Mapeamento:** trocar os gestos que controlam volume e reverberação, preservando o comportamento de silêncio quando falta deteção.
- **Música e áudio:** experimentar escalas, gama de notas e suavização; descrever o efeito auditivo.
- **Desempenho:** comparar com e sem visualização, registando FPS e tempo de inferência; explicar os limites dessas medições.
- **Testes:** propor casos para distância mínima/máxima, valores fora do intervalo, medida zero e mão ausente.
- **Extensão:** acrescentar um controlo baseado noutro ponto da mão e definir como o testar.
- No final, os alunos devem conseguir explicar o percurso completo **imagem → dados → decisão → som**, identificar as bibliotecas envolvidas e justificar escolhas de calibração, interação e desempenho.
