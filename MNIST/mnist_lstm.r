library(keras)

mnist <- dataset_mnist()
x_eğitim <- mnist$train$x 
y_eğitim <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_eğitim <- array_reshape(x_eğitim, c(nrow(x_eğitim), 28, 28))
x_eğitim <- x_eğitim / 255
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28))
x_test <- x_test / 255

sınıf_sayısı <- length(unique(y_eğitim))
y_eğitim <- to_categorical(y_eğitim, sınıf_sayısı)
y_test <- to_categorical(y_test, sınıf_sayısı)

yığın_sayısı <- 128
unite_sayısı <- 256
resim_boyutu <- c(28, 28)
devir_sayısı <- 20

model <- keras_model_sequential(name = "Temel_LSTM_Modeli")
model %>%
  layer_lstm(name = "LSTM_Katman", 
             units = unite_sayısı, 
             input_shape = resim_boyutu ) %>% 
  layer_dense(name = "Cikis_Katmani", 
              units = sınıf_sayısı, 
              activation = "softmax")

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

summary(model)

model %>% fit(
  x_eğitim, y_eğitim,
  batch_size = yığın_sayısı,
  epochs = devir_sayısı
)

sonuçlar <- model %>% evaluate(x_test, y_test, 
                               batch_size = yığın_sayısı)
cat('Test verisi kaybı:', sonuçlar[[1]], '\n')
cat('Test verisi doğruluğu:', sonuçlar[[2]], '\n')