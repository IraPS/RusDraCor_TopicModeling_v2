# Export image with width=1200 and height=800

library('ggplot2')

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

data <- read.csv('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/rusdracor_topic_modeling/graphs_6_topics/by_year/year_probs_for_R.csv', sep=';')

data_topic0 <- data[data$Topic=='Topic0', ]
data_topic1 <- data[data$Topic=='Topic1', ]
data_topic2 <- data[data$Topic=='Topic2', ]
data_topic3 <- data[data$Topic=='Topic3', ]
data_topic4 <- data[data$Topic=='Topic4', ]
data_topic5 <- data[data$Topic=='Topic5', ]

num_years_topic0 <- nrow(data_topic0)
num_years_topic1 <- nrow(data_topic1)
num_years_topic2 <- nrow(data_topic2)
num_years_topic3 <- nrow(data_topic3)
num_years_topic4 <- nrow(data_topic4)
num_years_topic5 <- nrow(data_topic5)

topic0_graph <- ggplot(data = data_topic0, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Семья 1") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic0-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                              axis.title=element_text(size=20,face="bold"))

topic1_graph <- ggplot(data = data_topic1, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Любовь") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic1-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                             axis.title=element_text(size=20,face="bold"))

topic2_graph <- ggplot(data = data_topic2, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Семья 2") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic2-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                               axis.title=element_text(size=20,face="bold"))

topic3_graph <- ggplot(data = data_topic3, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Двор") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic3-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                            axis.title=element_text(size=20,face="bold"))

topic4_graph <- ggplot(data = data_topic4, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Деньги") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic4-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                              axis.title=element_text(size=20,face="bold"))

topic5_graph <- ggplot(data = data_topic5, aes(x=Year, y=Probability)) + geom_line() + labs(x='', y="Люди и события") + theme(legend.position="none") + scale_y_continuous(limits=c(0,100)) + stat_smooth(aes(y=Probability, x=Year), formula = y ~ s(x, k = num_years_topic5-1), method = "gam", se = FALSE) + theme(axis.text=element_text(size=12),
                                                                                                                                                                                                                                                                                                                      axis.title=element_text(size=20,face="bold"))

multiplot(topic0_graph , topic1_graph , topic2_graph , topic3_graph , topic4_graph, topic5_graph, cols=2)



