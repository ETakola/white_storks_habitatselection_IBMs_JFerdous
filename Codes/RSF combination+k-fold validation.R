setwd("D:/4th Semester/thesis/models/RSF")
library(tidyverse)
library(glmmTMB)     
library(car)         
library(performance) 
library(terra)   
library(sjPlot)
library(pROC)
library(dplyr)
library(broom.mixed)
library(ggplot2)
library(ggpubr)

d<-readRDS("filtered_data.rds")
d
d$LULC_class <- as.factor(d$LULC_class)
d$LULC_class


levels(d$LULC_class) <- c(
  "10"  = "Rainfed cropland",
  "11"  = "Herbaceous cover cropland",
  "130" = "Grassland",
  "182" = "Marsh",
  "190" = "Impervious surface",
  "20"  = "Irrigated cropland",
  "210" = "Water body",
  "51"  = "Open EG BrL Forest",
  "61"  = "Open DC BrL Forest",
  "62"  = "Closed DC BrL Forest",
  "71"  = "Open EG NeedleL Forest",
  "72"  = "Closed EG NeedleL Forest",
  "81"  = "Open DC NeedleL Forest",
  "82"  = "Closed DC NeedleL Forest"
)
d$LULC_class <- relevel(d$LULC_class, ref = "Closed EG NeedleL Forest")
#class(d$LULC_class)
#library(lme4)
#rsf1_glmer <- glmer(Used ~ LULC_class + (1 | bird_id), family = binomial, data = d)
#summary(rsf1_glmer)
#fit using optimizer for singularity problem
fit1<- Used ~ LULC_class + (1| bird_id) 
rsf1<-glmmTMB(fit1, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim, 
                                       optArgs=list(method="BFGS")))
ranef(rsf1)$cond$bird_id
summary(rsf1)
###glmmTMB(formula, data = your_data, family = your_family,
###control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))
#NDVI
fit2<- Used ~ NDVI+ (1 +NDVI | bird_id)
rsf2<-glmmTMB(fit2, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf2)
ranef(rsf2)$cond$bird_id

#NDWI
fit3<- Used ~ NDWI+ (1 + NDWI | bird_id)
rsf3<-glmmTMB(fit3, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf3)
ranef(rsf3)$cond$bird_id

#shei
fit4<- Used ~ shei_scaled+ (1 | bird_id)
rsf4<-glmmTMB(fit4, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf4)

#shdi
fit5<- Used ~ shdi_scaled+ (1 | bird_id)
rsf5<-glmmTMB(fit5, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf5)
ranef(rsf5)$cond$bird_id


#pd
fit6<- Used ~ pd_scaled+ (1 | bird_id)
rsf6<-glmmTMB(fit6, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf6)

#ent
fit7<- Used ~ ent_scaled+ (1 | bird_id)
rsf7<-glmmTMB(fit7, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf7)
#ed
fit8<- Used ~ ed_scaled+ (1 | bird_id)
rsf8<-glmmTMB(fit8, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf8)


####Multi variate models####
######NDVI combis######
#NDVI+ed
fit9<- Used ~ NDVI + ed_scaled+ (1 + NDVI | bird_id)
rsf9<-glmmTMB(fit9, family = binomial, data = d,
              control = glmmTMBControl(optimizer=optim,
                                       optArgs=list(method="BFGS")))
summary(rsf9)

#NDVI+shdi
fit10<- Used ~ NDVI  + shdi_scaled+ (1 + NDVI | bird_id)
rsf10<-glmmTMB(fit10, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf10)

#NDVI+shei
fit11<- Used ~ NDVI  + shei_scaled+ (1 + NDVI | bird_id)
rsf11<-glmmTMB(fit11, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf11)

#NDVI+ent
fit12<- Used ~ NDVI  + ent_scaled+ (1 +NDVI | bird_id)
rsf12<-glmmTMB(fit12, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf12)

#NDVI+pd
fit13<- Used ~ NDVI + pd_scaled+ (1 + NDVI | bird_id)
rsf13<-glmmTMB(fit13, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf13)

#NDVI+LULC_class
fit13A<- Used ~ NDVI+LULC_class + (1 +NDVI | bird_id)
rsf13A<-glmmTMB(fit13A, family = binomial, data = d,
                control = glmmTMBControl(optimizer=optim,
                                         optArgs=list(method="BFGS")))
summary(rsf13A)
#NDVI+LULC_class+ed
fit14<- Used ~ NDVI + LULC_class + ed_scaled+ (1 + NDVI | bird_id)
rsf14<-glmmTMB(fit14, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf14)

#NDVI+LULC_class+shdi
fit15<- Used ~ NDVI + LULC_class + shdi_scaled+ (1 + NDVI | bird_id)
rsf15<-glmmTMB(fit15, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf15)

#NDVI+LULC_class+shei
fit16<- Used ~ NDVI + LULC_class + shei_scaled+ (1 + NDVI | bird_id)
rsf16<-glmmTMB(fit16, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf16)

#NDVI+LULC_class+ent
fit17<- Used ~ NDVI + LULC_class + ent_scaled+ (1 +NDVI | bird_id)
rsf17<-glmmTMB(fit17, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf17)

#NDVI+LULC_class+pd
fit18<- Used ~ NDVI + LULC_class + pd_scaled+ (1 +NDVI | bird_id)
rsf18<-glmmTMB(fit18, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf18)

#NDVI+ed+shei
fit19<- Used ~ NDVI + ed_scaled+ shei_scaled+ (1 +NDVI | bird_id)
rsf19<-glmmTMB(fit19, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf19)

fit20<- Used ~ NDVI +LULC_class + ed_scaled +shei_scaled+ (1 +NDVI | bird_id)
rsf20<-glmmTMB(fit20, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf20)


######NDWI combis######
#NDWI+ed
fit21<- Used ~ NDWI + ed_scaled+ (1 +NDWI | bird_id)
rsf21<-glmmTMB(fit21, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf21)

#NDWI+shdi
fit22<- Used ~ NDWI + shdi_scaled+ (1 +NDWI | bird_id)
rsf22<-glmmTMB(fit22, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf22)

#NDWI+shei
fit23<- Used ~ NDWI + shei_scaled+ (1+ NDWI | bird_id)
rsf23<-glmmTMB(fit23, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf23)

#NDWI+ent
fit24<- Used ~ NDWI + ent_scaled+ (1+ NDWI | bird_id)
rsf24<-glmmTMB(fit24, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf24)

#NDWI+pd
fit25<- Used ~ NDWI + pd_scaled+ (1+ NDWI | bird_id)
rsf25<-glmmTMB(fit25, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf25)

#NDWI+LULC
fit26<- Used ~ NDWI + LULC_class+ (1+ NDWI | bird_id)
rsf26<-glmmTMB(fit26, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf26)

#NDWI+LULC_class+ed
fit27<- Used ~ NDWI + LULC_class + ed_scaled+ (1 +NDWI | bird_id)
rsf27<-glmmTMB(fit27, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf27)

#NDWI+LULC_class+shdi
fit28<- Used ~ NDWI + LULC_class + shdi_scaled+ (1 +NDWI | bird_id)
rsf28<-glmmTMB(fit28, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf28)

#NDWI+LULC_class+shei
fit29<- Used ~ NDWI + LULC_class+ shei_scaled+ (1+ NDWI |bird_id)
rsf29<-glmmTMB(fit29, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf29)
ranef(rsf29)$cond$bird_id

#NDWI+LULC_class+ent
fit30<- Used ~ NDWI + LULC_class + ent_scaled+ (1+ NDWI | bird_id)
rsf30<-glmmTMB(fit30, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf30)

#NDWI+LULC_class+pd
fit31<- Used ~ NDWI + LULC_class + pd_scaled+ (1+ NDWI | bird_id)
rsf31<-glmmTMB(fit31, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf31)

#NDWI+ed+shei
#fit32<- Used ~ NDWI + scale(ed)+scale(shei)+ (1 | bird_id)
fit32<-Used ~ NDWI+ ed_scaled+shei_scaled+( 1 + NDWI  | bird_id)
rsf32<-glmmTMB(fit32, family = binomial, data = d, na.action = na.exclude,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf32)

#NDWI+LULC_class+ed+shei
fit33<-Used ~ NDWI+ LULC_class+ ed_scaled+shei_scaled+( 1 + NDWI  | bird_id)
rsf33<-glmmTMB(fit33, family = binomial, data = d, na.action = na.exclude,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf33)

#LULC+ed
fit34<- Used ~ LULC_class + ed_scaled+ (1 | bird_id)
rsf34<-glmmTMB(fit34, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf34)

#LULC+shdi
fit35<- Used ~ LULC_class + shdi_scaled+ (1 | bird_id)
rsf35<-glmmTMB(fit35, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf35)

#LULC+shei
fit36<- Used ~ LULC_class + shei_scaled+ (1 | bird_id)
rsf36<-glmmTMB(fit36, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf36)

#LULC+ent
fit37<- Used ~ LULC_class + ent_scaled+ (1 | bird_id)
rsf37<-glmmTMB(fit37, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf37)

#LULC+pd
fit38<- Used ~ LULC_class + pd_scaled+ (1 | bird_id)
rsf38<-glmmTMB(fit38, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf38)

#LULC+ed+pd
fit39<- Used ~ LULC_class +ed_scaled+ pd_scaled+ (1 | bird_id)
rsf39<-glmmTMB(fit39, family = binomial, data = d,
               control = glmmTMBControl(optimizer=optim,
                                        optArgs=list(method="BFGS")))
summary(rsf39)

####All model output as Txt####
models <- list(
  rsf1 = rsf1,rsf2 = rsf2,rsf3 = rsf3,rsf4 = rsf4,rsf5 = rsf5,
  rsf6 = rsf6,rsf7 = rsf7,rsf8 = rsf8,rsf9 = rsf9,rsf10= rsf10,
  rsf11=rsf11, rsf12=rsf12, rsf13=rsf13, rsf13A=rsf13A, rsf14=rsf14,
  rsf15=rsf15, rsf16=rsf16, rsf17=rsf17, rsf18=rsf18, rsf19=rsf19,rsf20=rsf20, 
  rsf21=rsf21, rsf22=rsf22, rsf23=rsf23, rsf24=rsf24,rsf25=rsf25, 
  rsf26=rsf26, rsf27=rsf27, rsf28=rsf28, rsf29=rsf29, rsf30=rsf30, 
  rsf31=rsf31, rsf32=rsf32, rsf33=rsf33, rsf34=rsf34, rsf35=rsf35,
  rsf36=rsf36, rsf37=rsf37, rsf38=rsf38, rsf39=rsf39)

sink("All_RSF_models_summary_03Nov25.txt")

for (name in names(models)) {
  cat("\n\n==============================\n")
  cat("Model:", name, "\n")
  cat("==============================\n\n")
  print(summary(models[[name]]))
}
sink()

####k-fold####
library(groupdata2)

set.seed(123)
d$bird_id<-as.factor(d$bird_id)
# Group wise folds
d <- groupdata2::fold(d, k = 5, id_col = "bird_id")

cv_results <- data.frame(Fold = integer(), AUC = numeric(), Accuracy = numeric())

for (i in 1:5) {
  cat("Running fold", i, "...\n")
  
  train_data <- d %>% filter(.folds != i)
  test_data  <- d %>% filter(.folds == i)
  
  
  model<-glmmTMB(Used ~ NDWI + LULC_class + ed_scaled + shei_scaled + (1 + NDWI | bird_id),
                 family = binomial, data = train_data, na.action = na.exclude,
                 control = glmmTMBControl(optimizer=optim,
                                          optArgs=list(method="BFGS")))
  
  test_data$pred <- predict(model, newdata = test_data, type = "response", allow.new.levels = TRUE)
  test_data$Used <- as.numeric(as.character(test_data$Used))
  test_data$Used[test_data$Used > 1] <- 1
  test_data$Used[is.na(test_data$Used)] <- 0
  
  # Compute AUC and Accuracy
  roc_obj <- roc(test_data$Used, test_data$pred)
  auc_val <- as.numeric(auc(roc_obj))
  acc_val <- mean(ifelse(test_data$pred > 0.5, 1, 0) == test_data$Used, na.rm = TRUE)
  
  cv_results <- rbind(cv_results, data.frame(Fold = i, AUC = auc_val, Accuracy = acc_val))
}

# Summarise mean +/- SD
cv_summary <- cv_results %>%
  summarise(mean_AUC = mean(AUC),
            sd_AUC   = sd(AUC),
            mean_Accuracy = mean(Accuracy),
            sd_Accuracy   = sd(Accuracy))

print(cv_results)
print(cv_summary)


d <- d %>% select(-starts_with(".folds"), everything())
d <- groupdata2::fold(d, k = 5, id_col = "bird_id", handle_existing_fold_cols = "remove")

roc_data_list <- list()
auc_values <- c()

# Cross-validation loop
for (i in 1:5) {
  cat("Computing ROC curve for fold", i, "...\n")
  
  # Split data
  train_data <- d %>% filter(.folds != i)
  test_data  <- d %>% filter(.folds == i)
  
  # Fit model on training data
  model <- glmmTMB(Used ~ NDWI + LULC_class + ed_scaled + shei_scaled + (1 + NDWI | bird_id),
                   family = binomial,
                   data = train_data,
                   control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))
  
  # Predict on test data
  test_data$pred <- predict(model, newdata = test_data, type = "response", allow.new.levels = TRUE)
  
  # Ensure Used is numeric 0/1
  test_data$Used <- as.numeric(as.character(test_data$Used))
  test_data$Used[test_data$Used > 1] <- 1
  test_data$Used[is.na(test_data$Used)] <- 0
  
  # Compute ROC
  roc_obj <- roc(test_data$Used, test_data$pred)
  auc_val <- as.numeric(auc(roc_obj))
  auc_values[i] <- auc_val
  
  # Extract ROC curve points
  roc_df <- data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities,
    Fold = paste0("Fold ", i, " (AUC=", round(auc_val, 3), ")")
  )
  roc_data_list[[i]] <- roc_df
}

# Combine and plot
roc_all <- bind_rows(roc_data_list)

ROC<-ggplot(roc_all, aes(x = FPR, y = TPR, color = Fold)) +
  geom_line(linewidth=1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Fold-wise ROC Curves for RSF33",
    subtitle = paste0("Mean AUC = ", round(mean(auc_values), 3),
                      " ± ", round(sd(auc_values), 3)),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Fold"
  ) +
  coord_equal() +
  theme(legend.position = "bottom")
ggsave("ROC.png", width = 10, height = 5, dpi = 500)




