library('ggplot2')

nouns_300_100 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/Only Nouns 300-100.csv', sep=';')

nouns_400_100 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/Only Nouns 400-100.csv', sep=';')

nouns_500_100 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/Only Nouns 500-100.csv', sep=';')

nouns_600_150 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/Only Nouns 600-150.csv', sep=';')

pos_400_100 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/POS restriction 400-100.csv', sep=';')

pos_500_100 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/POS restriction 500-100.csv', sep=';')

pos_600_150 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/POS restriction 600-150.csv', sep=';')

pos_700_200 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/POS restriction 700-200.csv', sep=';')

pos_900_300 <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/semdensity_2/POS restriction 900-300.csv', sep=';')

data <- rbind(nouns_300_100, nouns_400_100, nouns_500_100, nouns_600_150,
              pos_400_100, pos_500_100, pos_600_150, pos_700_200, pos_900_300)

data

ggplot(data = data, aes(x=numtopics, y=average_topic_semdensity_for_10_topwords)) + geom_line(aes(colour=model)) + scale_x_continuous(breaks = seq(4, 15, 1)) + labs(x="Количество тем", y="Семплотность", colour='Модель')
