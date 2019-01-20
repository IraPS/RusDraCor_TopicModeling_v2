gender_dist <- matrix(c(19.89, 25.64, 7.03, 16.36, 13.55, 17.54, 12.32, 15.9, 10.25, 23.56, 15.7, 22.27, 15.82, 14.52, 12.84, 30.21, 14.59, 12.02),ncol=6,byrow=TRUE)

rownames(gender_dist) <- c("Женский","Мужской","Неизвестно")

colnames(gender_dist) <- c("Семья 1",'Любовь', 'Семья 2', 'Двор', 'Деньги', 'Люди')

barplot(gender_dist, main="", xlab="", ylab="Вероятность", col=c("pink","blue", "grey"), beside=TRUE)

legend("topright", legend=rownames(gender_dist), cex=1, fill=c("pink","blue", "grey"))