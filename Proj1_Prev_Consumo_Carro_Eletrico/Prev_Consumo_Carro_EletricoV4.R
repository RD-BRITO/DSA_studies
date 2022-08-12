# Projeto referente ao curso Big Data Analytics com R e Microsoft Azure
# Machine Learning, da Formação Cientista de Dados da DSA-Data Science Academy

# Defininndo o diretório de trabalho

setwd("C:/FDC/BigDataRAzure/Projetos_Feedback/P1_Prev_Consumo_Carros")
getwd()
# Carregar pacotes

#install.packages("dplyr")
#install.packages("readxl")
#install.packages("Amelia")
#install.packages("corrplot")
#install.packages("randomForest")
#install.packages("ggplot2")

library("dplyr")
library("readxl")
library("Amelia")
library("corrplot")
library("randomForest")
library("ggplot2")
# ----------------------------------------------------------------------------
# Abaixo o roteiro que será seguido neste projeto.

# Etapa 1: Análise Exploratória

# 1. Entender o problema de negócio
# 2. Entender o dataset avaliando o dicionário de dados
# 3. Renomear as variáveis (se necessário)
# 4. Avaliar dados incompletos
# 5. Definir tratamento para os dados missing
# 6. Avaliar variáveis (correlação) e categóricas (tabelas)
# 7. Ajustar base de dados com as variáveis de interesse

# Etapa 2: Criação e avaliação do modelo de machine learning

# 1. Selecionar dados para treino e teste
# 2. Feature Selection: seleção das variáveis mais relevantes
# 3. Definir e implementar modelo de M.L.
# 4. Avaliar desempenho do modelo
# 5. Conclusão da análise

#----------------------------------------------------------------------------

## ETAPA 1 - ANÁLISE EXPLORATÓRIA


# 1. Entender o problema de negócio

# Objetivo do projeto:

# Usando um dataset com dados reais disponíveis publicamente, construir um modelo
# de Machine Learning capaz de prever o consumo de energia de carros elétricos 
# com base em diversos fatores, tais como o tipo e número de motores elétricos do
# veículo, o peso do veículo, a capacidade de carga, entre outros atributos.

# Ferramentas utilizadas: linguagem R

#----------------------------------------------------------------------------

# 2. Entender o dataset avaliando o dicionário de dados

# >> Não foi localizado um dicionário de dados na fonte. As variáveis foram
# >> avaliadas nominalmente e pesquisadas pontualmente para entendimento.

# >> Atenção especial ao parâmetro "Range (WLTP) [km]", referente à distância
# >> máxima percorrida com uma carga completa da bateria, sendo um dos parâmetros
# >> avaliados no teste WLTP (Worldwide Harmonised Light Vehicles Test Procedure).

# Obter base de dados

dataset_orig<-read_xlsx("FEV-data-Excel.xlsx")

# Exploração inicial
str(dataset_orig)
View(dataset_orig)
summary(dataset_orig)

#----------------------------------------------------------------------------
# 3. Renomear as variáveis, se necessário

# Renomear as variáveis removendo os colchetes das unidades de medida
nomes_orig<- names(dataset_orig)
nomes_orig

nomes <- names(dataset_orig)
nomesnovos<- c("Nom.", "Fab.","Modl.", "Preço_Mín._PNL", "Pot._kw", "Torque_Máx_Nm",
               "Tip.Freio","Tração","Cap.Bateria_kwh", "Range_WLTP_km", "Eixos_cm", "Compr._cm",
               "Larg._cm","Altura_cm","Peso_Mín_Vazio_kg","Peso_Bruto_Permitido_kg",
               "Carga_Máx_kg","Assentos","Portas","Tam_Pneus_in",
               "Vel_Máx_kph","Bagageiro_L","Aceleração_s","Carga_DC_Máx_kw",
               "Consumo_Méd_kWh_100km")

# Atribuir os novos nomes a cada posição do vetor de nomes

for (i in 1:length(nomes)) {
  nomes[i]<-nomesnovos[i]
}
rm(i)
nomes

# Alterar nomes no dataset

colnames(dataset_orig)<- nomes
View(dataset_orig)
rm(nomes)
rm(nomesnovos)

# ---------------------------------------------------------------------------
# 4. Avaliar dados incompletos

# Verificar variáveis com valores NA: Missmap

missmap(dataset_orig,
        legend = T,
        col = c("orange", "black"),
        main = "Variáveis com valores NA",
        rank.order = F,
        y.labels = seq(1:53),
        y.at = seq(53:1))


# >> Existem 6 variáveis com dados NA, sendo a maioria na variável de interesse
# >> "consumo médio".

# Avaliar % de linhas incompletas

sum(complete.cases(dataset_orig))
sum(!complete.cases(dataset_orig))

prop_linhas_comp<-(sum(complete.cases(dataset_orig))/(sum(complete.cases(dataset_orig))+
                                                   sum(!complete.cases(dataset_orig))))*100
prop_linhas_comp
rm(prop_linhas_comp)

# ----------------------------------------------------------------------------

# 5. Definir tratamento para os dados missing

# >> 79% das linhas do dataset estão completas.
# >> DECISÃO: remover as linhas com valores NA.

# Remover linhas com valores NA

dataset_V1 <- na.omit(dataset_orig)
sum(is.na(dataset_V1))
View(dataset_V1)

# ----------------------------------------------------------------------------

# 6. Avaliar variáveis (correlação) e categóricas (tabelas)

dev.off()
ind_var_num<-sapply(dataset_V1,is.numeric)
var_num <- dataset_V1[ind_var_num]
summary(var_num)

# Possíveis outliers: "Pot", "Torque", "Carga Máx", "Carga DC Máx"
boxplot(dataset_V1$`Pot._kw`)
boxplot(dataset_V1$`Torque_Máx_Nm`)
boxplot(dataset_V1$`Carga_DC_Máx_kw`)
boxplot(dataset_V1$`Carga_Máx_kg`)

# DECISÂO: Manter os valores dos parâmetros 

correlacao<- cor(var_num)

corrplot(correlacao)

rm(correlacao)
rm(ind_var_num)
rm(var_num)

# Verificação das variáveis categóricas

table(dataset_V1$Fab.)
table(dataset_V1$Tip.Freio)
table(dataset_V1$Tração)


# ----------------------------------------------------------------------------

# 7. Ajustar base de dados com as variáveis de interesse

# >> Inicialmente serão consideradas todas as variáveis do dataset, para posterior
# >> refinamento do modelo.

# Normalização dos dados

# Definir função Min-Max para normalização

normalizar<- function(x){
  if (is.numeric(x)==TRUE) {
x<-(x - min(x)) / (max(x) - min(x))
  }else{
x<-x  }
}

# Aplicar função ao dataset

datasetNorm<-as.data.frame(lapply(dataset_V1, normalizar))

View(datasetNorm)
str(datasetNorm)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# ETAPA 2: CRIAÇÃO E AVALIAÇÃO DO MODELO DE MACHINE LEARNING

# 1. Selecionar dados para treino e teste

set.seed(1012)
sample <- sample(c(T,F), nrow(datasetNorm), replace = T, prob = c(0.7,0.3))

treino <- datasetNorm[sample,]
teste<-datasetNorm[!sample,]
rm(sample)

dim(datasetNorm)
dim(treino)
dim(teste)

# >> 70% dos dados para treino e 30% para teste

# ----------------------------------------------------------------------------
# 2. Feature Selection: seleção das variáveis mais relevantes

# >> Aplicar o modelo RamdomForest para avaliar as variáveis

var_mod<-randomForest(Consumo_Méd_kWh_100km~.,data = datasetNorm, ntree = 100, nodesize = 10, importance = T)

varImpPlot(var_mod)

# >> Serão selecionadas inicialmente para uma análise com modelo de regressão 
# >> linear todas as variáveis presentes no dataset_V1 para comparar a relevância
# >> dos itens com a indicação do modelo RandomForest

# ----------------------------------------------------------------------------

# 3. Definir e implementar modelo de M.L.

# Será implementado um modelo de Regressão Linear

# Modelo 1: Adjusted R-squared:  NaN
M1 <-lm(Consumo_Méd_kWh_100km~., data = treino)
summary(M1)

# >> O modelo M1 não gerou resultados satisfatórios. Serão retiradas algumas variáveis 
# >> para a próxima versão do modelo.

# Modelo 2: Adjusted R-squared:   0.9354
# >> Mudanças: desconsiderar as variáveis abaixo por serem parâmetros tecnicamente
# >> pouco relevantes para o objeto de estudo.
# >> "Nom.","Fab.","Modl.","Preço_Mín._PNL", "Assentos", "Portas","Bagageiro_L"   

treino2<-treino%>%select(!c("Nom.","Fab.","Modl.","Preço_Mín._PNL", "Assentos", "Portas","Bagageiro_L"))

M2 <-lm(Consumo_Méd_kWh_100km~., data = treino2)
summary(M2)

# Modelo 3: Adjusted R-squared:  0.9554
# >> Mudanças: serão retiradas do "treino2" as variáveis com valor Pr(>|t|) acima de 0.5:
# >> "Aceleração_s","Carga_DC_Máx_kw","Altura_cm","Compr._cm","Larg._cm","Eixos_cm","Torque_Máx_Nm"

treino3<-treino2%>%select(!c("Aceleração_s",
                             "Carga_DC_Máx_kw",
                             "Altura_cm",
                             "Compr._cm",
                             "Larg._cm",
                             "Eixos_cm",
                             "Torque_Máx_Nm"))


M3 <-lm(Consumo_Méd_kWh_100km~., data = treino3)
summary(M3)

# Modelo 4: Adjusted R-squared:  0.9577
# >> Mudanças: retirada as variáveis com Pr(>|t|) acima de 0.5: 
# >> "Peso_Mín_Vazio_kg" e "Peso_Bruto_Permitido_kg" em relação ao dataset treino3.

treino4<-treino3%>%select(!c("Peso_Mín_Vazio_kg","Peso_Bruto_Permitido_kg"))

M4 <-lm(Consumo_Méd_kWh_100km~., data = treino4)
summary(M4)

# Modelo 5: Adjusted R-squared:  0.9552
# >> Mudanças: retirada a variável "Vel_Máx_kph" em relação aos dados de "treino4"

treino5<-treino4%>%select(!"Vel_Máx_kph")

M5 <-lm(Consumo_Méd_kWh_100km~., data = treino5)
summary(M5)

# Resumo:
# Modelo 1: Adjusted R-squared: NaN  
# Modelo 2: Adjusted R-squared: 0.9354 
# Modelo 3: Adjusted R-squared: 0.9554
# Modelo 4: Adjusted R-squared: 0.9577 <----
# Modelo 5: Adjusted R-squared: 0.9552

# >> Será utilizado o modelo M4 nos testes.

# >> Cálculo dos erros do modelo M4

# SSE

sse <- sum((fitted(M4) - treino4$Consumo_Méd_kWh_100km)^2)
sse

# SSR

ssr <- sum((fitted(M4) - mean(treino4$Consumo_Méd_kWh_100km))^2)
ssr

# SST

sst <- ssr + sse
sst

# R²

R_Squared <- ssr / sst
R_Squared # R² = 0.9699843

# ---------------------------------------------------------------------------
# 4. Avaliar desempenho do modelo

# >> Ajustando os dados de teste conforme os dados do treino4
teste4<-teste%>%select(!c("Nom.",
                          "Fab.",
                          "Modl.",
                          "Preço_Mín._PNL",
                          "Assentos",
                          "Portas",
                          "Bagageiro_L",
                          "Aceleração_s",
                          "Carga_DC_Máx_kw",
                          "Altura_cm",
                          "Compr._cm",
                          "Larg._cm",
                          "Eixos_cm",
                          "Torque_Máx_Nm",
                          "Peso_Mín_Vazio_kg",
                          "Peso_Bruto_Permitido_kg"))

Previsões <- predict(M4,teste4)

resultados<-cbind(teste4[9],Previsões)
names(resultados)<-c("Real","Previsto")
View(resultados)

# >> Cálculo dos erros do resultado do teste do modelo M4

# SSE

sse <- sum((resultados$Previsto - resultados$Real)^2)
sse

# SSR

ssr <- sum((resultados$Previsto - mean(resultados$Real))^2)
ssr

# SST

sst <- ssr + sse
sst

# R²

R_Squared <- ssr / sst
R_Squared # R² = 0.9233075

# --------------------------------------------------------------------------
# Gráfico
ggplot(resultados, aes(x = Real,y = Previsto), main = ) + 
  geom_point() + stat_smooth() + ggtitle("Resultado da Previsão do Modelo M4")

# ---------------------------------------------------------------------------

# 5. Conclusão da análise inicial

# >> Por meio do modelo M4 observou-se que 92,33% da variação no "Consumo Médio"
# >> pode ser explicado por 8 variáveis ("Pot._kw","Tip.Freio", "Tração", "Cap.Bateria_kwh",
# >> "Range_WLTP_km","Carga_Máx_kg", "Tam_Pneus_in" e "Vel_Máx_kph").   

# ---------------------------------------------------------------------------