library(dplyr)
library(ggplot2)
library(tidytext)
library(ggforce)
df <- read.csv("nll_output.csv")

# Define shapes
shape_values <- c("Qwen1.5" =20, "Yi" =20, "DeepSeek" =20, "InternLM2" =20, "Baichuan2" =20, "XVERSE" =20, "Llama3" = 2, "Mistral" = 2)

model_type <- c("Qwen1.5-7B" = "Qwen1.5", "Yi-6B" = "Yi", "deepseek-llm-7b-base" = "DeepSeek", "internlm2-7b" = "InternLM2", "Baichuan2-7B-Base" = "Baichuan2", 
"XVERSE-7B" = "XVERSE", "Meta-Llama-3-8B" = "Llama3", "Mistral-7B-v0.3" = "Mistral")

aspect_ratio <- 0.8
color_values <- c("blue", "red", "green", "purple", "pink", "orange", "darkgreen",  "brown")
names(color_values) <- names(shape_values)

df %>% 
filter(lang_code!="uig_Arab") %>%
group_by(model_name, lang_category, lang_code, lang_name) %>% 
summarise(mean_nll_sum = mean(nll_sum, na.rm = TRUE)) %>%
ungroup() %>%
mutate(model=model_type[model_name], lang_name=reorder_within(lang_name, -mean_nll_sum, lang_category, fun=mean)) %>%
ggplot(aes(x = mean_nll_sum, y = lang_name)) +
  geom_point(size = 3, aes(colour = model, shape=model)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  scale_y_reordered() +
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "Chinese Han Dialects (Other)", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Mean NLL Across Sentences", size=10) +
  scale_x_reverse()

ggsave("figures/exp1_boxplot.pdf", height = 7 , width = 7 * aspect_ratio)

df <- read.csv("bele_output.csv")

df %>% 
mutate(model=model_type[model_name], lang_name=reorder_within(lang_name, accuracy, lang_category, fun=mean))%>% 
 ggplot(aes(x = accuracy, y = lang_name)) +
  geom_point(size = 3, aes(colour = model, shape=model)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  geom_vline(xintercept = 0.25, linetype = "dashed", color = "black") +
  scale_y_reordered()+
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Accuracy") 
ggsave("figures/exp2_boxplot.pdf", height = 7 , width = 7 * aspect_ratio)