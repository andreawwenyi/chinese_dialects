library(dplyr)
library(ggplot2)
library(tidytext)
library(ggforce)
df <- read.csv("floresp_nll_output_v2.csv")

# Define shapes
shape_values <- c("Qwen2.5" =20, "Yi" =20, "DeepSeek-R1-Llama" =20, "DeepSeek-R1-Qwen"=20,
"InternLM2" =20, "Baichuan2" =20, "XVerse" =20, "Llama3" = 2, "Mistral" = 2, "Olmo2"=2, "Gemma2"=2)


aspect_ratio <- 0.8
color_values <- c("blue", "red", "darkgreen",  "green", "grey", "pink", "orange",  "purple", "brown", "black", "cyan")
names(color_values) <- names(shape_values)

df %>% 
group_by(model_name_abrev, lang_category, lang_code, lang_name) %>% 
summarise(mean_nll_sum = mean(nll_sum, na.rm = TRUE)) %>%
ungroup() %>%
mutate(lang_name=reorder_within(lang_name, -mean_nll_sum, lang_category, fun=mean)) %>%
ggplot(aes(x = mean_nll_sum, y = lang_name)) +
  geom_point(size = 3, aes(colour = model_name_abrev, shape=model_name_abrev)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  scale_y_reordered() +
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "Chinese Han Dialects (Other)", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Mean NLL Across Sentences", size=10) +
  scale_x_reverse()

ggsave("figures/exp1_boxplot.pdf", height = 7 , width = 7 * aspect_ratio)

df %>% 
group_by(model_name_abrev, lang_category, lang_code, lang_name) %>% 
summarise(mean_ip = mean(ip_base_eng_Latn, na.rm = TRUE)) %>%
ungroup() %>%
mutate(lang_name=reorder_within(lang_name, mean_ip, lang_category, fun=mean)) %>%
ggplot(aes(x = mean_ip, y = lang_name)) +
  geom_point(size = 3, aes(colour = model_name_abrev, shape=model_name_abrev)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  scale_y_reordered() +
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "Chinese Han Dialects (Other)", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Mean IP Across Sentences", size=10)

ggsave("figures/exp1_boxplot_ip_base_eng.pdf", height = 7 , width = 7 * aspect_ratio)

df %>% 
group_by(model_name_abrev, lang_category, lang_code, lang_name) %>% 
summarise(mean_ip = mean(ip_base_cmn_Hans, na.rm = TRUE)) %>%
ungroup() %>%
mutate(lang_name=reorder_within(lang_name, mean_ip, lang_category, fun=mean)) %>%
ggplot(aes(x = mean_ip, y = lang_name)) +
  geom_point(size = 3, aes(colour = model_name_abrev, shape=model_name_abrev)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  scale_y_reordered() +
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "Chinese Han Dialects (Other)", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Mean IP Across Sentences", size=10) 

ggsave("figures/exp1_boxplot_ip_base_cmn.pdf", height = 7 , width = 7 * aspect_ratio)

df %>% 
group_by(model_name_abrev, lang_category, lang_code, lang_name) %>% 
summarise(mean_ppl = mean(ppl, na.rm = TRUE)) %>%
ungroup() %>%
mutate(lang_name=reorder_within(lang_name, -mean_ppl, lang_category, fun=mean)) %>%
ggplot(aes(x = mean_ppl, y = lang_name)) +
  geom_point(size = 3, aes(colour = model_name_abrev, shape=model_name_abrev)) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
  scale_y_reordered() +
  facet_col(factor(lang_category, levels = c("Mandarin Chinese", "Chinese Han Dialects (Other)", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
  scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
  scale_color_manual(values = color_values, breaks=names(shape_values)) +
  labs(y="Languages", colour="Models", shape="Models", x="Mean PPL Across Sentences", size=10) +
  scale_x_reverse()

ggsave("figures/exp1_boxplot_ppl.pdf", height = 7 , width = 7 * aspect_ratio)

# df <- read.csv("bele_output.csv")

# df %>% 
# mutate(lang_name=reorder_within(lang_name, accuracy, lang_category, fun=mean))%>% 
#  ggplot(aes(x = accuracy, y = lang_name)) +
#   geom_point(size = 3, aes(colour = model_name_abrev, shape=model_name_abrev)) +
#   theme_minimal() +
#   theme(panel.grid.major.y = element_blank(), axis.title=element_text(size=10), legend.text = element_text(size=8)) +
#   geom_vline(xintercept = 0.25, linetype = "dashed", color = "black") +
#   scale_y_reordered()+
#   facet_col(factor(lang_category, levels = c("Mandarin Chinese", "US/European", "Northeast Asian", "Southeast Asian", "Chinese Ethnic Minorities")) ~ ., scales = "free_y", space = "free") +
#   scale_shape_manual(values = shape_values, breaks=names(shape_values)) +
#   scale_color_manual(values = color_values, breaks=names(shape_values)) +
#   labs(y="Languages", colour="Models", shape="Models", x="Accuracy") 
# ggsave("figures/exp2_boxplot.pdf", height = 7 , width = 7 * aspect_ratio)