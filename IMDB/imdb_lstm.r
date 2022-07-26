library(keras)

maks_metin <- 2495
imdb <- dataset_imdb(maxlen = maks_metin) 
x_eğitim <- imdb$train$x 
y_eğitim <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y
x_eğitim <- pad_sequences(x_eğitim, maxlen = maks_metin) 
x_test   <- pad_sequences(x_test,   maxlen = maks_metin)

yığın_sayısı <- 128
devir_sayısı <- 25
unite_sayısı <- 64
çıkış_sayısı <- 1
model <- keras_model_sequential(name = "Temel_LSTM_Modeli") 
model %>% 
  layer_embedding(name = "Metin_Giris_Katmani",
                  input_dim = maks_metin, output_dim = unite_sayısı) %>%
  layer_lstm(name = "LSTM_Katmani", units = unite_sayısı) %>%  
  layer_dense(name = "Cikis_Katmani", 
              units = çıkış_sayısı, activation = "sigmoid")

model %>% compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics = c("acc"))

summary(model)
model %>% fit(x_eğitim, y_eğitim,
              epochs = devir_sayısı,
              batch_size = yığın_sayısı)

sonuçlar <- model %>% evaluate(x_test, y_test, batch_size = yığın_sayısı)
cat('Test verisi kaybı:', sonuçlar[[1]], '\n')
cat('Test verisi doğruluğu:', sonuçlar[[2]], '\n')