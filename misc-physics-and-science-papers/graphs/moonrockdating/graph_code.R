library(ggplot2)

df <- read.csv("apollo17_moon_rock_data.csv")

fit <- lm(Sr87toSr86 ~ Rb87toSr86, data = df)

slope <- coef(fit)[2]
slope_se <- summary(fit)$coefficients[2, "Std. Error"]
r2 <- summary(fit)$r.squared

slope_label <- sprintf(
  "slope == %.2f %%+-%% %.2f %%*%% 10^{-2}",
  slope * 100,
  slope_se * 100
)

r2_label <- sprintf("r^2 == %.4f", r2)

p <- ggplot(df, aes(x = Rb87toSr86, y = Sr87toSr86)) +
  geom_point(size = 2.8, color = "#1f77b4") +
  geom_smooth(method = "lm", se = FALSE, color = "#1f77b4", linewidth = 1) +
  annotate(
    "text",
    x = 0.085, y = 0.7087,
    label = slope_label,
    parse = TRUE, size = 6
  ) +
  annotate(
    "text",
    x = 0.084, y = 0.7062,
    label = r2_label,
    parse = TRUE, size = 6
  ) +
  labs(
    title = "Rubidium Strontium Isochron: Lunar Rocks",
    x = expression(frac({}^{87}*Rb, {}^{86}*Sr)),
    y = expression(frac({}^{87}*Sr, {}^{86}*Sr))
  ) +
  scale_x_continuous(
    limits = c(0, 0.17),
    breaks = seq(0, 0.16, by = 0.02)
  ) +
  scale_y_continuous(
    limits = c(0.6925, 0.7168),
    breaks = seq(0.695, 0.715, by = 0.005)
  ) +
  theme_classic(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

p

ggsave("rb_sr_isochron_plot.png", plot = p)