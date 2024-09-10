library(ggplot2)
library(dplyr)
library(ggrepel)
langs <- read.csv("langs.csv") %>% 
mutate(Speakers=as.numeric(Speakers))
# ----- exp 1 data ----- #
exp1_data <- read.csv("nll_output.csv") %>% 
group_by(model_name, lang_category, lang_code, lang_name) %>% 
summarise(mean_nll_sum = mean(nll_sum, na.rm = TRUE)) %>%
ungroup() %>%
mutate(is_chinese_model = if_else(grepl("Mistral|Meta", model_name), "International Models", "Chinese Models")) %>%
group_by(is_chinese_model, lang_code, lang_name, lang_category) %>%
summarise(mean_score = mean(mean_nll_sum, na.rm = TRUE)) %>%
ungroup() %>%
mutate(exp="Unnormalized PPL")%>% 
left_join(langs, by=c("lang_code" = "lang_code_flores", "lang_name", "lang_category")) %>%
filter(lang_code!="yue_Hant", lang_code != "uig_Arab")

# ----- exp 2 data ----- #
exp2_data <- read.csv("bele_output.csv") %>% mutate(is_chinese_model = if_else(grepl("Mistral|Meta", model_name), "International Models", "Chinese Models")) %>%
group_by(is_chinese_model, lang_code, lang_name, lang_category) %>%
summarise(mean_score = mean(accuracy, na.rm = TRUE)) %>%
ungroup() %>%
mutate(exp="Zero-shot MC Accuracy")%>% 
left_join(langs, by=c("lang_code" = "lang_code_bele", "lang_name", "lang_category"))

data <- exp1_data %>%
  bind_rows(exp2_data)

data %>%
ggplot(aes(x = GDP..billion.USD., y = mean_score)) +
  geom_point(shape = 4, size = 2) + 
  geom_text_repel(aes(label = lang_name)) + 
  facet_grid(exp~is_chinese_model, scales = "free_y") +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA, size = 1)) +
  geom_smooth(method = "lm", se = FALSE, aes(group = is_chinese_model, color = is_chinese_model), size=0.7) +
  labs(
    x = "National GDP (billion USD)",
    y = "Mean Metric Across Models"
  ) +
  scale_x_log10() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),  
    legend.position="none"
  ) +
  scale_color_manual(values = c("coral", "cornflowerblue"))

ggsave("./figures/scatter_gdp.pdf", height=7, width=10)
  
data %>%
ggplot(aes(x = Speakers, y = mean_score)) +
  geom_point(shape = 4, size = 2) + 
  geom_text_repel(aes(label = lang_name)) + 
  facet_grid(exp~is_chinese_model, scales = "free_y") +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA, size = 1)) +
  geom_smooth(method = "lm", se = FALSE, aes(group = is_chinese_model, color = is_chinese_model), size=0.7) +
  labs(
    x = "# of Language Speakers",
    y = "Mean Metric Across Models"
  ) +
  scale_x_log10() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),  
    legend.position="none"
  ) +
  scale_color_manual(values = c("coral", "cornflowerblue"))

ggsave("./figures/scatter_speakers.pdf", height=7, width=10)

# data %>% 
# ggplot(aes(x = GDP..billion.USD., y = mean_nll_sum)) +
#   geom_point(shape = 4, size = 3) +  # Plot the points with square shape
#   geom_text_repel(aes(label = lang_name)) + 
#   facet_wrap(~is_chinese_model, scales = "free_x") +
#   theme_bw() + 
#   geom_smooth(method = "lm", se = FALSE, aes(group = is_chinese_model, color = is_chinese_model), size=0.7)+
#   labs(
#     x = "GDP (billion USD)",
#     y = "Mean Unnormalized PPL Across Models",
#   ) + 
#   scale_x_log10() + 
#   theme(
#     strip.text = element_text(size = 12, face = "bold"), 
#     legend.position="none"
#   ) +
#   scale_color_manual(values = c("coral", "cornflowerblue"))

# ggsave("./figures/exp1_GDP.pdf", height=5, width=10)


# # ----- exp 2 ----- #



# data %>%
# ggplot(aes(x = Speakers, y = mean_accuracy)) +
#   geom_point(shape = 4, size = 3) +  # Plot the points with square shape
#   geom_text_repel(aes(label = lang_name)) + 
#   facet_wrap(~is_chinese_model, scales = "free_x") +
#   theme_bw() + 
#   geom_smooth(method = "lm", se = FALSE, aes(group = is_chinese_model, color = is_chinese_model), size=0.7) +
#   labs(
#     x = "# of Speakers",
#     y = "Mean Accuracy Across Models"
#   ) +
#   scale_x_log10() +
#   theme(
#     strip.text = element_text(size = 12, face = "bold"),  
#     legend.position="none"
#   ) +
#   scale_color_manual(values = c("coral", "cornflowerblue"))

# ggsave("./figures/exp2_speakers.pdf", height=5, width=10)


# data %>% 
# ggplot(aes(x = GDP..billion.USD., y = mean_accuracy)) +
#   geom_point(shape = 4, size = 3) +  # Plot the points with square shape
#   geom_text_repel(aes(label = lang_name)) + 
#   facet_wrap(~is_chinese_model, scales = "free_x") +
#   theme_bw() + 
#   geom_smooth(method = "lm", se = FALSE, aes(group = is_chinese_model, color = is_chinese_model), size=0.7)+
#   labs(
#     x = "GDP (billion USD)",
#     y = "Mean Accuracy Across Models",
#   ) + 
#   scale_x_log10() + 
#   theme(
#     strip.text = element_text(size = 12, face = "bold"), 
#     legend.position="none"
#   ) +
#   scale_color_manual(values = c("coral", "cornflowerblue"))

# ggsave("./figures/exp2_GDP.pdf", height=5, width=10)
