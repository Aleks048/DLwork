#fit OLS for dri with predictions
def OLSregression(funOnPredicted,data,numOfClasses:int,useThrechold:bool,numEpochs:int):

    
    y = []
    Xlist = []

    for k in data:
        y.append(data[k][0])
        Xlist.append(funOnPredicted(data[k][1:][0],useThrechold))#[0] since wrapped into the array


    X = numpy.asarray(Xlist)

    #normalizing
    #X = (X-numpy.mean(X,axis = 0))/numpy.std(X,axis=0)
   
    input_x = Input(shape=(numOfClasses,))
    lin = Dense(1)(input_x)

    #keras model
    #model = keras.Model(inputs = input_x,outputs = lin)

    #model.summary()

    #optimizer = keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False)

    #model.compile(loss="mse",optimizer=optimizer)
    #model.fit(X, y, epochs = numEpochs, batch_size = 50, verbose = 1)
    #weights = model.get_weights()

    #sklearn model
    reg =sklearn.linear_model.HuberRegressor(epsilon=1.01,max_iter=100000,alpha=0.000001,fit_intercept=False,tol = 1e-40).fit(X,y)
    #reg =sklearn.linear_model.BayesianRidge(fit_intercept=False,tol = 1e-20).fit(X,y)
   
    weights = [reg.coef_]


    b = weights[0]
    return b
