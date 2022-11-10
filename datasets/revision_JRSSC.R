##
## plot for JRSSC paper
##
## voir <tidyverse.org>
## 
library(tidyverse)
setwd("/Users/thithanhyen/AResearch/wtot_coclust_match/datasets")
## ####
## mRNA
## ####
colnames <- c("ID", names(read_delim("LFC_Striatum_mrna.txt", delim = "\t", n_max = 1)))
dat <- read_delim("LFC_Striatum_mrna.txt", delim = "\t", skip = 1)
colnames(dat) <- colnames
##
DAT <- dat %>% 
  pivot_longer(!ID, names_to = c("polyQ_length", "time"), values_to = "value", names_pattern = "at_Q([0-9]+)_time_([0-9]+)") %>%
  mutate(polyQ_length = as.integer(polyQ_length),
         time = as.integer(time)) %>%
  mutate(polyQ_length_time = paste(polyQ_length, time)) %>%
  group_by(polyQ_length_time) %>%
  mutate(mean = mean(value), sd = sd(value)) %>%
  mutate(n = 1:n()) %>%
  ungroup
DAT %>%  ggplot() +
  stat_ecdf(aes(x = value)) +
  geom_line(aes(x = value, y = pnorm(value, mean, sd)), col = "red", alpha = 0.5) +
  facet_wrap(vars(time, polyQ_length), nrow = 3, labeller = "label_both") +
  theme(strip.background = element_blank(), strip.placement = "outside") +
  xlab("gene expression") +
  ylab("empirical cdf") +
  xlim(c(-1, 1)) +
  ggtitle("mRNA")
DAT %>%  ggplot() +
  geom_density(aes(x = value)) +
  facet_wrap(vars(time, polyQ_length), nrow = 3, labeller = "label_both") +
  theme(strip.background = element_blank(), strip.placement = "outside") +
  xlab("gene expression") +
  ylab("kernel density estimate") +
  xlim(c(-0.1, 0.1)) +
  ggtitle("mRNA")
## #####
## miRNA
## #####
colnames <- c("ID", names(read_delim("LFC_Striatum_mirna.txt", delim = "\t", n_max = 1)))
dat <- read_delim("LFC_Striatum_mirna.txt", delim = "\t", skip = 1)
colnames(dat) <- colnames
##
DAT <- dat %>% 
  pivot_longer(!ID, names_to = c("polyQ_length", "time"), values_to = "value", names_pattern = "at_Q([0-9]+)_time_([0-9]+)") %>%
  mutate(polyQ_length = as.integer(polyQ_length),
         time = as.integer(time)) %>%
  mutate(polyQ_length_time = paste(polyQ_length, time)) %>%
  group_by(polyQ_length_time) %>%
  mutate(mean = mean(value), sd = sd(value)) %>%
  mutate(n = 1:n()) %>%
  ungroup
DAT %>%  ggplot() +
  stat_ecdf(aes(x = value)) +
  geom_line(aes(x = value, y = pnorm(value, mean, sd)), col = "red", alpha = 0.5) +
  facet_wrap(vars(time, polyQ_length), nrow = 3, labeller = "label_both") +
  theme(strip.background = element_blank(), strip.placement = "outside") +
  xlab("gene expression") +
  ylab("empirical cdf") +
  xlim(c(-1, 1)) +
  ggtitle("miRNA")
DAT %>%  ggplot() +
  geom_density(aes(x = value)) +
  facet_wrap(vars(time, polyQ_length), nrow = 3, labeller = "label_both") +
  theme(strip.background = element_blank(), strip.placement = "outside") +
  xlab("gene expression") +
  ylab("kernel density estimate") +
  xlim(c(-0.1, 0.1)) +
  ggtitle("miRNA")

##
