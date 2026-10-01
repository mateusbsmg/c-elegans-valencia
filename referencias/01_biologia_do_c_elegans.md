# Biologia do *C. elegans* usada no projeto

## Sistema nervoso (conectoma)
- Hermafrodita adulto: **302 neurônios**. Na matriz de Cook et al. 2019 usada aqui: 83 sensoriais, 83 interneurônios, 108 motores, 20 da faringe e 8 da postura de ovos (HSN, VC).
- Ligações do arquivo `dados/conectoma.json`: 3.709 pares com sinapse química, 1.093 junções comunicantes (gap junctions) e 956 ligações neurônio→músculo.
- A maioria dos neurônios não dispara "spikes". Eles usam **potencial graduado**, por isso o modelo usa taxa contínua.
- Neurotransmissores: acetilcolina e glutamato (em geral excitam), GABA (inibe: DD, VD, RME, RIS, AVL, DVB…), além de monoaminas e peptídeos (serotonina, dopamina, octopamina, tiramina, FMRFamida).
- Fontes:
  - Cook et al. 2019, *Nature* 571:63 ("Whole-animal connectomes of both C. elegans sexes"), SI 5;
  - White et al. 1986 ("The Mind of a Worm");
  - Varshney et al. 2011;
  - dados via [openworm/c302](https://github.com/openworm/c302) e [CElegansNeuroML](https://github.com/openworm/CElegansNeuroML).

## Músculos e locomoção
- **95 músculos da parede do corpo** em 4 feixes: dorsal esquerdo e direito (24 cada), ventral esquerdo (23) e ventral direito (24). Músculo só **contrai**: dobrar para um lado é contrair um lado e relaxar o outro.
- O verme rasteja **deitado de lado**. Por isso a onda dorsal/ventral fica no plano da placa: no nosso 2D, "dorsal" e "ventral" viram os dois lados da curva.
- 75 motoneurônios no cordão ventral:
  - AS (11), DA (9), DB (7), DD (6);
  - VA (12), VB (11), VC (6), VD (13).
- Papéis de cada classe:
  - B (DB/VB): andar para frente.
  - A (DA/VA): andar para trás.
  - D (DD/VD): GABA, a "inibição cruzada" (quando um lado contrai, o outro relaxa).
- Comandos: **AVB/PVC** para frente, **AVA/AVD/AVE** para trás (ré).
- **Propriocepção:** os motoneurônios B sentem a curvatura do trecho vizinho, e é isso que faz a onda viajar da cabeça para a cauda (Wen et al. 2012, *Neuron*).
- Números de rastejar no ágar:
  - onda de ~0,3 a 0,5 Hz;
  - comprimento de onda de ~0,6 a 0,7 corpo;
  - velocidade de ~0,14 a 0,2 mm/s;
  - corpo de ~1 mm.
  - Nadando, a frequência é maior e a onda mais longa.
- Física: no ágar o arrasto de lado é de 20 a 40 vezes maior que ao longo do corpo (*resistive force theory*). É o que transforma a onda em avanço.
- Fontes:
  - [Scientific Reports 2021, modelo de conectoma e locomoção](https://www.nature.com/articles/s41598-021-92690-2);
  - [Wen et al. 2012 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/23177960/);
  - [Fang-Yen et al. 2010, PNAS](https://www.pnas.org/doi/full/10.1073/pnas.1003016107);
  - [Inhibition underlies fast undulatory locomotion (eNeuro)](https://www.eneuro.org/content/8/2/ENEURO.0241-20.2020);
  - [arXiv 1702.04988, biomecânica](https://arxiv.org/pdf/1702.04988).

## Quimiotaxia (achar comida pelo cheiro)
- **AWA** e **AWC** detectam odores voláteis:
  - o AWA é ativado quando o odor aumenta;
  - o AWC é ativado quando o odor **some** ("OFF").
- **ASE** detecta substâncias dissolvidas: ASEL responde a aumento, ASER a queda.
- Duas estratégias:
  - **klinocinese / piruetas:** se o cheiro piora, reverte e muda de direção. O AIB promove reversões.
  - **klinotaxia:** vai curvando a cabeça para o lado onde o cheiro é mais forte (o AIY participa da curva).
- O verme compara o cheiro **no tempo** (dC/dt) enquanto a cabeça balança.
- Fontes:
  - [Nature Communications 2018, AWA/AWC](https://www.nature.com/articles/s41467-018-05151-2);
  - [Cell Reports 2015, circuito da subida do gradiente](https://www.cell.com/cell-reports/fulltext/S2211-1247(15)00917-1);
  - [Genetics 2021, transdução quimiossensorial](https://academic.oup.com/genetics/article/217/3/iyab004/6162992).

## Dor, prazer e fome
- **Dor (nocicepção):** o **ASH** é o nociceptor polimodal (toque no nariz, químicos, osmolaridade). Ele leva à ré via AVA. FLP, ALM, AVM e PLM respondem ao toque.
- **Prazer / comida:**
  - neurônios de dopamina (CEP, ADE, PDE) sentem as bactérias sob o corpo e o verme desacelera;
  - serotonina (NSM, ADF) ao engolir comida faz o verme ficar e comer.
- **Fome:** a **octopamina** (RIC) sinaliza fome e faz o verme "explorar" (roaming). A serotonina tem o efeito oposto (dwelling).
- Fontes:
  - [ASH e ASI, Nature Comms 2014](https://www.nature.com/articles/ncomms6655);
  - [Dopamina sensibiliza a fuga, EMBO J 2011](https://link.springer.com/article/10.1038/emboj.2011.22);
  - [Serotonina e octopamina antagonistas, J Neurosci 2017](https://www.jneurosci.org/content/37/33/7811);
  - [WormBook: aminas biogênicas](http://www.wormbook.org/chapters/www_monoamines/monoamines.html).

## Quanto tempo vive sem comer
- Vida normal do adulto: ~2 a 3 semanas.
- Sem comida:
  - a larva L1 recém-nascida sobrevive ~2 semanas;
  - a larva **dauer** (estágio de resistência) sobrevive **meses** (~4) com a gordura guardada.
- Os **90 dias** do projeto são, portanto, plausíveis para um verme em estado dauer.
- Fonte: [Starvation Responses Throughout the C. elegans Life Cycle, Genetics 2020](https://academic.oup.com/genetics/article/216/4/837/6065822).

## OpenWorm
- Projeto aberto (licença MIT) que simula o verme inteiro: c302 (rede neural em NeuroML) e Sibernetic (corpo 3D).
- É muito mais detalhado e pesado que este projeto: uma simulação de 15 ms leva minutos.
- Aqui usamos só os **dados** dele.
- Links: [github.com/openworm/OpenWorm](https://github.com/openworm/OpenWorm), [c302 (artigo)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6158223/).
