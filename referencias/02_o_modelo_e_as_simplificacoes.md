# Como a simulação funciona (e o que foi simplificado)

## Relógio e arena
- **1 dia de vida = 2 s do corpo.** A 1×, 90 dias ≈ 3 minutos.
- Arena de 8 × 5 mm. O verme nasce do lado esquerdo e a mancha de bactérias fica do lado direito.
- Cada mancha dá 60 "dias de comida". Quando acaba, nasce outra do lado oposto.
- Cheiro: um gradiente gaussiano (alcance ~3 mm) em volta da comida.

## Corpo (`simulacao/mundo.py`)
- 24 segmentos, um por fileira de músculos, ao longo de 1 mm.
- Os músculos dorsais e ventrais de cada fileira definem a curvatura. Ela segue com um atraso de ~0,1 s (músculo e elasticidade).
- A física usa **resistive force theory**: arrasto de lado 20× maior que ao longo do corpo. A cada passo, calcula-se o movimento do corpo inteiro que deixa força e torque totais iguais a zero.
- Validação: uma onda de 0,5 Hz dá ~0,2 mm/s, perto do verme real no ágar.

## Impulsos (a recompensa)
- **Fome:** cresce exponencialmente com os dias sem comer, `(e^(d/15) − 1)/(e^(90/15) − 1)`. Vale 0 logo após comer e 1 no dia 90, quando o verme morre.
- **Prazer:**
  - comer, e vale mais quanto maior a fome;
  - o cheiro ficar mais forte. Esse termo é uma diferença, então ir e voltar não rende nada: só chegar perto rende.
- **Dor:**
  - encostar na parede (nariz dói mais);
  - o cheiro piorar;
  - o incômodo da fome;
  - morrer.
- Recompensa = prazer − dor. É o que os dois aprendizados maximizam.

## Cérebro A: conectoma (`simulacao/cerebro_conectoma.py`)
- 302 neurônios de potencial graduado, com as ligações reais (Cook 2019).
- O sinal de cada sinapse vem do neurotransmissor: ACh e glutamato excitam, GABA inibe, tiramina inibe (RIM→AVB).
- A força da rede é calibrada para não "convulsionar" (raio espectral 0,9).
- Músculo = soma das sinapses neuromusculares.
- Sensores ligados aos neurônios de verdade:
  - AWA, AWC, ASE: cheiro;
  - ASH, FLP: dor no nariz;
  - ALM, AVM, PLM, PVM: toque;
  - RIC: fome;
  - NSM, ADF: comendo;
  - CEP, ADE, PDE: sobre a comida.
- **Simplificações assumidas:**
  - **Propriocepção:** os motoneurônios B sentem o trecho à frente da sua área, e os A o trecho de trás. O ganho depende do comando (AVB ou AVA).
  - **Osciladores:** um na cabeça (para frente) e um na cauda (para trás), acionados por AVB e AVA. A origem exata do ritmo no verme ainda é debatida.
  - **Neuromodulação resumida:** a octopamina acelera; a serotonina e a dopamina desaceleram.
  - **Limiares** dos motoneurônios (ventral −0,4, dorsal −0,8), ajustados para a marcha sair equilibrada.
- **Aprendizado (estratégias evolutivas, OpenAI-ES):**
  - A cada geração vivem 33 vermes: o cérebro atual mais 16 pares de variações (+/−) nos pesos.
  - Cada par tem o mesmo começo e o mesmo ruído, para a comparação ser justa.
  - Os pesos andam na direção das variações que sentiram mais prazer e menos dor.
  - Ajusta **3.818 parâmetros**: sinapses que chegam em neurônios sensoriais, interneurônios e da cabeça, mais o viés deles.
  - O circuito da marcha no corpo e as junções neuromusculares ficam fixos, como instinto. No primeiro teste, deixar tudo livre desregulava o andar.
  - Roda na CPU: ~11 s por geração na velocidade máxima.

## Cérebro B: Laya (`simulacao/cerebro_laya.py`)
- A cada 1 s do corpo, o Laya lê um texto com fome, nível do cheiro, de que lado da cabeça o cheiro é mais forte, toque, se está comendo e o efeito da ação anterior.
- Ele escolhe entre 5 ações:
  - seguir em frente;
  - virar para o lado dorsal;
  - virar para o lado ventral;
  - pirueta;
  - parar e comer.
- Um **gerador de ritmo (CPG)** transforma a ação na onda muscular. Os parâmetros dele também são aprendidos: **139 no total**.
  - Por ação (5 × 4): frequência, força da onda, comprimento de onda e curva da cabeça.
  - Por músculo (95): um ganho, ou seja, quanto aquele músculo contrai com o mesmo comando.
  - Por fileira (24): um atraso de fase, isto é, o "tempo" daquele trecho na onda.
  - Como aprende: por estratégias evolutivas dentro do mesmo treino. Os 16 vermes formam 8 pares com variações +/− nos músculos e o mesmo ponto de partida. As variações que somaram mais prazer − dor (chegar mais rápido e mais certeiro) puxam os parâmetros.
  - Ficam salvos em `runs/laya_rl/motor.npz` e, junto com o Laya, em `atual/` e `melhor/` (`motor.npy`).
  - No 2D, esquerdo e direito do mesmo lado (DL/DR, VL/VR) somam na mesma curva.
- **Aprendizado (REINFORCE com linha de base do grupo):**
  - 16 vermes vivem 30 dias sorteando ações pelas chances do Laya.
  - Depois, cada ação ganha ou perde chance conforme o prazer − dor que veio depois, comparado com a média do grupo no mesmo instante.
  - Não há professor.
  - Roda na placa de vídeo, com pausa acima de 80 °C.
- Testes de referência com políticas escritas à mão (32 vermes, 30 dias):

  | Política | Vermes que acharam comida |
  |---|---|
  | Sempre em frente | 2 de 32 |
  | Aleatória | 1 de 32 |
  | "Oráculo" que usa bem o texto | 18 de 32 |

  Ou seja, o texto contém a informação necessária.

## Linha de base (antes de qualquer treino)
- Conectoma original: 3 de 32 vermes acharam a comida em 30 dias.

## Reflexos e ajustes do cérebro B (fora do Laya)
- **Transição suave entre ações:** frequência, força e curva da onda mudam em ~0,5 s. Cada verme começa num ponto diferente da onda.
- **Largada:** todos na mesma região, direção parecida e mesma fome, mas não no mesmo milímetro.
- **Reflexo de fuga:** ao bater o nariz, dá ré por 0,8 s na hora (ASH/FLP → AVA; Kaplan & Horvitz 1993).
- **Desaceleração sobre a comida** (*basal slowing*, dopamina): mais forte com fome (*enhanced slowing*, serotonina; Sawin, Ranganathan & Horvitz 2000).
- **Exploração pela fome** (octopamina): longe da comida e com fome, a onda fica até 40% mais rápida.

## Genes sensoriais (cérebro B)
- Além dos 139 genes musculares, cada verme tem **6 genes sensoriais**, cada um ligado ou desligado:
  - fome (RIC);
  - nível do cheiro (AWA);
  - lado do cheiro, isto é, a klinotaxia (AIY);
  - toque (ASH/ALM);
  - "está comendo" (NSM);
  - memória da última ação (AIB).
- Só os sentidos expressos entram no texto que o Laya lê. Por isso os textos têm tamanhos diferentes de verme para verme.
- **Custo:** cada sentido expresso custa um pouco de energia, cerca de 1 ponto de valência negativa por episódio de 30 dias. A evolução pesa: vale a pena manter este sentido?
- **Herança:** cruzamento sentido a sentido. A mutação liga ou desliga um sentido com 8% de chance.
- **Fundadores:** o F0 tem todos os sentidos; os outros sorteiam cada sentido com 70% de chance. O selvagem sorteia com 60%.

## Bactérias (a comida)
- 30 a 50 microcolônias espalhadas, que se movem devagar (8 a 30 µm/s), somem ao ser comidas e renascem em outro lugar depois de 2 a 6 s.
- **As três espécies:**

  | Espécie | Nutrição (dias) | Cheiro | Efeito |
  |---|---|---|---|
  | *Comamonas* | 6 | 1,0 | o melhor alimento |
  | *Bacillus* | 4 | 0,6 | bom, cheiro fraco |
  | *Pseudomonas* | 3 | 1,5 | atrai, mas causa mal-estar |

  - Fontes: *Comamonas* acelera o desenvolvimento (Watson et al. 2014, *Cell*). A *Pseudomonas* é patogênica e o verme real aprende a evitá-la (Zhang, Lu & Bargmann 2005, *Nature*).
- **Mal-estar:** comer *Pseudomonas* causa valência negativa que se dissipa em ~4 s. Com fome extrema, ainda compensa comê-la.
- **Disputa:** no treino do cérebro B, os 16 vermes dividem a mesma placa, e o que um come some para os outros.
- **Texto do Laya:** diz qual espécie domina o cheiro ("mostly Pseudomonas") e qual acabou de comer.

## Energia e seleção dos genes musculares
- **Custo do ATP:** cada passo custa `0,0025 × contração média dos 95 músculos`. Um jeito de se mexer mais eficiente sai na frente.
- **Eficiência** = mm percorridos pelo nariz ÷ contração total. Aparece em laranja no gráfico.
- **Nota da elite:** cada genoma é julgado pela média de até 5 vidas. Quem ganhou por sorte cai, e quem é bom de verdade se firma.

## Revisão do cérebro B (após o diagnóstico da geração 50)
- **Diagnóstico:** o teste de discernimento mostrou que o Laya treinado dava ~20% para cada ação em qualquer situação. A diferenciação caiu de 17,9 para 0,9 ponto percentual. A causa foi o bônus de entropia alto (0,08).
- **Correção:**
  - o Laya recomeça do original;
  - bônus de entropia de 0,005;
  - **âncora KL** (β = 0,1) em direção ao Laya original, que fica congelado na placa em meia precisão;
  - só as 8 camadas de cima (20 a 27) e a cabeça treinam, cerca de 125 M parâmetros. O pico de memória caiu de ~10 GB para ~4,4 GB.
- **Teste de discernimento automático** a cada 10 episódios, mostrado na página: 7 situações controladas e o quanto as chances mudam entre elas.
- **Coordenação motora fina:** 7 ações (frente, virar leve ou forte para cada lado, pirueta, parar e comer). O genoma passou a ter 147 genes musculares, e os genomas antigos foram convertidos. A cabeça balança menos (60% da onda no nariz), e a curva de direção se desfaz suavemente pelo pescoço.
- **Movimentos bruscos:** custo por "salto" de comando voluntário, `0,08 × Σ|Δ(velocidade da onda, 3×curva, 3×ômega)|`. Por episódio: sempre em frente 0; suave 1,5; brusco 10. O reflexo de fuga não é cobrado.
- **Morte:** valência negativa máxima do modelo (30). Nenhum outro evento chega perto.
- **Ilhas:** 24 vermes em 3 ilhas (A, B, C) de 8. Por ilha há 2 elites, 5 filhos e 1 selvagem. A cada 5 gerações, o melhor de cada ilha migra para a seguinte.
- **Registro da valência por episódio** em `historico.json`, campo `componentes`: comer, cheiro melhorando ou piorando, parede, fome, infecção, morte, energia, sentidos e movimentos bruscos.

## Tribos e feromônio
- **Tribos:** cada selvagem que nasce funda uma tribo nova, com cor própria (ângulo de ouro). Só o próprio selvagem aparece em branco.
- **Regra da diversidade:** se sobrar uma única tribo (fora os selvagens), entram 2 vermes de tribos novas no lugar dos 2 piores filhos. As elites nunca são substituídas.
- **7º gene sensorial, "feromônio" (ascarosídeos; neurônios ASK/ADL).** Quem o expressa lê:
  - a **lotação:** soma do feromônio dos outros vermes no nariz, com alcance de 0,8 mm;
  - a **escassez:** quantas microcolônias há num raio de 1,2 mm.
- O feromônio só existe quando os vermes dividem a placa (treino do cérebro B). Na população salva, o gene foi sorteado com 50% de chance.

## Cheiro por espécie e memória de infecção
- O gene "lado do cheiro" passou a dizer o lado de cada espécie presente, por exemplo: *"Comamonas smell is stronger on the dorsal side; Pseudomonas smell is stronger on the ventral side."* Isso segue a biologia: os odores de cada bactéria ativam receptores diferentes em AWA e AWC.
- O gene "memória" acrescenta *"It remembers getting sick after eating Pseudomonas."* depois de uma infecção no episódio (aversão aprendida; Zhang, Lu & Bargmann 2005).
- **Limitação observada:** comer é automático, e a *Pseudomonas* está espalhada entre as outras bactérias. Por isso, desviar dela pela direção pouco reduz o quanto ela é comida. No teste com um oráculo, a proporção comida ficou entre 24% e 38%, contra 25% na placa.

## Hall da fama
- O melhor genoma de todos os tempos (média de pelo menos 3 vidas) fica guardado em `runs/laya_rl/hall_da_fama.*`.
- Se, numa geração, nenhuma elite supera a média dele, ele volta a viver no lugar do último filho da sua ilha. No placar aparece com ↺.
- Enquanto está vivo, a nota dele no hall acompanha a média das vidas mais recentes. Se era sorte, ele deixa de voltar; se um descendente for melhor, o descendente toma o lugar.
