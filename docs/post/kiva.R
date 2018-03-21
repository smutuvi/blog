library(dplyr)
library(ggplot2)
library(treemap)
library(gridExtra)
library(grid)
library(fmsb)
library(leaflet)
library(readr)
library(knitr)
library(kableExtra)
library(formattable)
library(plotly)

packageVersion('plotly')

loans_df <- read_csv('content/post/data/kiva/kiva_loans.csv')

loans_df %>% filter(country=="Kenya") %>% group_by(region) %>% summarise(nr = length(sector)) %>% top_n(20,wt=nr) %>% ungroup() %>%
  ggplot(aes(x = reorder(region,nr), y = nr)) +
  geom_bar(stat="identity", fill="lightblue", colour="black") +
  #geom_text(aes(label=nr), hjust=-0.2, position=position_dodge(width=0.6)) +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="", x ="Region in Kenya", y = "Number of loans") 


loans_df %>% group_by(sector) %>% summarise(nr = length(activity))  %>% ungroup() %>%
  ggplot(aes(x = reorder(sector,nr), y = nr)) +
  geom_bar(stat="identity", fill="lightblue", colour="black") +
  #geom_text(aes(label=nr), hjust=0.2, position=position_dodge(width=0.6)) +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="", x ="Sector (all)", y = "Number of loans") -> d1

loans_df %>% group_by(activity) %>% summarise(nr = length(sector)) %>% top_n(20,wt=nr) %>% ungroup() %>%
  ggplot(aes(x = reorder(activity,nr), y = nr)) +
  geom_bar(stat="identity", fill="gold", colour="black") +
  #geom_text(aes(label=nr), hjust=-0.2, position=position_dodge(width=0.6)) +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="", x ="Activities (top 20 by number of loans)", y = "Number of loans") -> d2

loans_df %>% group_by(country) %>% summarise(nr = length(sector)) %>% top_n(20,wt=nr) %>% ungroup() %>%
  ggplot(aes(x = reorder(country,nr), y = nr)) +
  geom_bar(stat="identity", fill="tomato", colour="black") +
  #geom_text(aes(label=nr), hjust=-0.2, position=position_dodge(width=0.6)) +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="", x ="Countries (top 20 by number of loans)", y = "Number of loans") -> d3

loans_df %>% group_by(region) %>% summarise(nr = length(sector)) %>% top_n(20,wt=nr) %>% ungroup() %>%
  ggplot(aes(x = reorder(region,nr), y = nr)) +
  geom_bar(stat="identity", fill="lightgreen", colour="black") +
  #geom_text(aes(label=nr), hjust=-0.2, position=position_dodge(width=0.6)) +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="", x ="Regions (top 20 by number of loans)", y = "Number of loans") -> d4

grid.arrange(d1, d2, d3, d4, ncol=2)

loans_df %>% filter(country=="Kenya") %>% group_by(sector, activity) %>% summarise(nr = length(region)) %>% top_n(5,wt=nr) %>% ungroup() %>%
  treemap(
    index=c("sector","activity"), 
    type="value",
    vSize = "nr",  
    vColor = "nr",
    palette = "RdBu",  
    title=sprintf("Loans per sector and activity"), 
    title.legend = "Number of loans",
    fontsize.title = 14 
  )

loans_df %>% group_by(region, sector) %>% summarise(nr = length(activity)) %>% top_n(100,wt=nr) %>% ungroup() %>%
  treemap(
    index=c("region","sector"), 
    type="value",
    vSize = "nr",  
    vColor = "nr",
    palette = "RdBu",  
    title=sprintf("Loans per country and region"), 
    title.legend = "Number of loans",
    fontsize.title = 14 
  )

# gender
strcount <- function(x, pattern, split){
  unlist(lapply(strsplit(x, split), function(z) na.omit(length(grep(pattern, z)))))
}
loans_df$nMale <- strcount(loans_df$borrower_genders, "^male", " ")
loans_df$nFemale = strcount(loans_df$borrower_genders, "female", " ")

loans_df$borrowers_gen = "Not specified"
loans_df$borrowers_gen[(loans_df$nMale != 0 & loans_df$nFemale == 0)] = "Male"
loans_df$borrowers_gen[(loans_df$nMale == 0 & loans_df$nFemale != 0)] = "Female"
loans_df$borrowers_gen[(loans_df$nMale != 0 & loans_df$nFemale != 0)] = "Female & Male"

loans_df %>% filter(country=="Kenya") %>% group_by(borrowers_gen) %>% summarise(nr = length(borrower_genders)) %>% ungroup() %>%
  ggplot(aes(x = reorder(borrowers_gen,nr), y = nr/1000)) +
  geom_bar(stat="identity", aes(fill=borrowers_gen), colour="black") +
  theme_bw(base_size = 12)  +
  labs(title="Number of loans/borrowers gender", x ="Gender", y = "Number of loans (thousands)", fill="Borrowers genders")

# distribution of female and male borrowers
loans_df %>% filter(country=="Kenya") %>% group_by(sector) %>% summarise(nrF = mean(nFemale))  %>% ungroup() %>%
  ggplot(aes(x = reorder(sector,nrF), y = nrF)) +
  geom_bar(stat="identity", fill="tomato", colour="black") +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="Average number of female borrowers/loan", subtitle="Loans with at least one female borrower", 
       x ="Sector", y = "Average number of female borrowers/loan") -> d1

loans_df %>% filter(country=="Kenya")  %>% group_by(sector) %>% summarise(nrM = mean(nMale))  %>% ungroup() %>%
  ggplot(aes(x = reorder(sector,nrM), y = nrM)) +
  geom_bar(stat="identity", fill="lightblue", colour="black") +
  coord_flip() + theme_bw(base_size = 10)  +
  labs(title="Average number of male borrowers/loan", subtitle="Loans with at least one male borrower",
       x ="Sector", y = "Average number of male borrowers/loan") -> d2

loans_df  %>% filter(country=="Kenya") %>% group_by(sector) %>% ungroup() %>%
  ggplot(aes(x=reorder(sector,nFemale), y= nFemale, col=sector)) + 
  geom_boxplot() +
  theme_bw() + coord_flip() + labs(x="Sector", y="Number of female borrowers/loan", col="Sector",
                                   title="Boxplot distribution | female borrowers/loan",
                                   subtitle="Grouped by sector") -> d3
loans_df %>% filter(country=="Kenya") %>% group_by(sector) %>% ungroup() %>%
  ggplot(aes(x=reorder(sector,nMale), y= nMale, col=sector)) + 
  geom_boxplot() +
  theme_bw() + coord_flip() + labs(x="Sector", y="Number of male borrowers/loan", col="Sector",
                                   title="Boxplot distribution | male borrowers/loan",
                                   subtitle="Grouped by sector") -> d4
grid.arrange(d1,d2,d3,d4,ncol=2)

