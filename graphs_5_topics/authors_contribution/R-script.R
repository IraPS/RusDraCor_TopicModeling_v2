library('ggplot2')
library(dplyr)
library(tidyr)

data <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/authors_contribution/joint_data.csv', stringsAsFactors = FALSE, sep=';')

data %>% separate(Author, c("Surname", "rest")) -> data

data <- data[,c('Surname')]
data <- as.table(table(unlist(data)))
data <- as.matrix(data)


data1 <- matrix(c(3,4,8,2,1,4,5,1,2,1,1,1,7,1,1,3,7,3,3,3,2,8,2,14,4), ncol=1, byrow=TRUE)
rownames(data1) <- c('Блок','Булгаков','Гоголь','Горький','Грибоедов','Гумилёв','Крылов','Лермонтов','Ломоносов','Маяковский','Найдёнов','Озеров','Островский','Писемский','Плавильщиков','Прутков','Пушкин','Сумароков','Сухово-Кобылин','Толстой A','Толстой Л','Тургенев','Фонвизин','Чехов','Шаховской')
colnames(data1) <- c('Num_of_plays')
data1 <- t(data1)
authors <- c('Блок','Булгаков','Гоголь','Горький','Грибоедов','Гумилёв','Крылов','Лермонтов','Ломоносов','Маяковский','Найдёнов','Озеров','Островский','Писемский','Плавильщиков','Прутков','Пушкин','Сумароков','Сухово-Кобылин','Толстой A','Толстой Л','Тургенев','Фонвизин','Чехов','Шаховской')

barplot(data1,main='Распределение пьес и авторов в корпусе', ylab='Количество пьес', las=2)
