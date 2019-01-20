library('ggplot2')

data <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/num_plays_per_year/plays_per_year.csv', sep=';')
numbers <- data$Num_of_plays
decades <- data$Decade

end_point = 0.5 + length(numbers) + length(numbers)-1
barplot(numbers, main="Распределение пьес по десятилетям", ylab="Количество пьес", names.arg=decades, las=2)