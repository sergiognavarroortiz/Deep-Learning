###Forecasting Vehicle Value

#Descargar Data
loca="C:/Users/Invest In data/Documents/TRABAJO/SMART DATA/R ELEMTOS DE AYUDA/Used Car Data Capitulo 12 DNN"
setwd(loca)
urlloc="http://www.amstat.org/publications/jse/datasets/kuiper.xls"
download.file(urlloc,destfile ="CarValue.xls",mode="wb")   #Por alguna razon no se pudo descargar.

##Evaluate the Relationship between used car price and other attributes

#Cargar Data
library(readxl)
dataset<-read_excel("CarValue.xls")

str(dataset)

#Vistazo de la variable a predecir (Price)
hist(dataset$Price,col = "blue",xlab = "Precio",ylab = "Frecuencia",main = "Histograma de Precio")

#Grafico de relacion entre variable Precio y Millas recorridas
plot(dataset$Price,dataset$Mileage,col="Green",xlab="Precio",ylab = "Millas Recorridas")

#Grafico de cajas de precio con respecto a la marca del coche y Tipo de coche (Sedan, coupe)
par(mfrow=c(2,1))
boxplot(dataset$Price~dataset$Make,col="Green",ylab="Precio")
boxplot(dataset$Price~dataset$Type,col="Darkgreen",ylab="Precio")

#Grafico de Valores de precio vs cilindros, puertas, control crucero, asientos en piel
par(mfrow=c(2,2))
plot(dataset$Price~dataset$Cylinder,col="red",ylab = "Precio ($)",xlab="Cilindros")
plot(dataset$Price~dataset$Doors,col="red",ylab = "Precio ($)",xlab="No. Puertas")
plot(dataset$Price~dataset$Cruise,col="red",ylab = "Precio ($)",xlab="Tiene control crucero")
plot(dataset$Price~dataset$Leather,col="red",ylab = "Precio ($)",xlab="Tiene asientos en piel")

##Preprocesamiento de datos

class(dataset$Make)

dataset<-as.data.frame(unclass(dataset))   #Este codigo convierte las variables de caracter en factores (Revisar despues si funciona para convertir las variables en la clase adecuada o solo a facrores)

class(dataset$Make)

head(dataset$Make)

#Conversion a matriz binarias que tengan una columna por nivel 
require(nnet)
Make=class.ind(dataset$Make)
Model=class.ind(dataset$Model)
Trim=class.ind(dataset$Trim)
Type=class.ind(dataset$Type)

head(Make)

#Unir las variables factores
car_fac<-scale(cbind(Make,Model,Trim,Type))

#Unir variables numericas
numericals<-cbind(dataset$Mileage,dataset[,7:12])

#Funcion de normalizacion para variables numericas
max_min_Range<-function(x){(x-min(x))/(max(x)-min(x))}

#Normalizacion
numericals<-max_min_Range(numericals)

#Creacion de la tabla ya lista para realizar el modelado
data<-as.matrix(cbind(car_fac,numericals,
                      log(dataset$Price)))

#Numero de columnas de la data
ncol(data)

#Creacion de la tabla de entrenamiento y de prediccion 
rand_seed=2016
set.seed(rand_seed)
n_train=700
train<-sample(1:nrow(data),n_train,FALSE)
x_train<-data[train,1:97]
y_train<-data[train,98]
x_test<-data[-train,1:97]
y_test<-data[-train,98]

##How to Benefit from Mini Batching

#Se contruira un modelo de 2 capas ocultas con un tamaño de lote de 200.

#Creacion del modelo
require(deepnet)
set.seed(rand_seed)
fit1<-nn.train(x=x_train,y=y_train,
               hidden = c(3,2),
               activationfun = "tanh",
               momentum = 0.15,
               learningrate = 0.85,
               numepochs = 200,
               batchsize = 100,
               output = "linear")

#Midiendo el desempeño del modelo
pred1<-nn.predict(fit1,x_train)
require(Metrics)

mse(pred1,y_train)
#Valor de 0.001809124

cor(pred1,y_train)^2
#Valor de 0.9893688

##Cross Validation

#k=10
require(caret)
cv_error<-NULL
k<-10
set.seed(rand_seed)
folds<-createFolds(train,k=k)

#Revisar los folds
folds$Fold02

#A Simple for loop
for (i in 1:k) {
  x_train_cv<-data[folds[[i]],1:97]
  y_train_cv<-data[folds[[i]],98]
  
  set.seed(rand_seed)
  fit_cv<-nn.train(x=x_train_cv,
                   y=y_train_cv,
                   hidden = c(3,2),
                   activationfun = "tanh",
                   momentum = 0.15,
                   learningrate = 0.85,
                   numepochs = 200,
                   batchsize =100,
                   output="linear")
  
  pred_cv<-nn.predict(fit_cv,x_train_cv)
  cv_error[i]=cor(pred_cv,y_train_cv)^2
}

#Error de primera iteracion
cv_error[1]

#Grafico de caja de los valores de error de los 10 ajustes de k-fold cross validation
par(mfrow=c(1,1))
boxplot(cv_error,horizontal = TRUE,col="Darkgreen")

#Performance on the Test Set
pred1_test<-nn.predict(fit1,x_test)

mse(pred1_test,y_test)
#Valor de 0.001480497

cor(pred1_test,y_test)^2
#Valor de 0.991796

#Grafico de relaciones de valor real de prediccion y el valor ajustado
plot(exp(pred1_test),col="white",ylab = "Precio ($)",xlab = "Observaciones")
lines(exp(y_test),col="Blue")
lines(exp(pred1_test),col="red")











