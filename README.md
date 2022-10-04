# Fast and efficient image novelty detection based on mean-shifts

Hermann, M., Umlauf, G., Goldlücke B., & Franz M. O. (2022). Fast and efficient image novelty detection based on mean-shifts. Sensors | Unusual Behavior Detection Based on Machine Learning.

Cite us
```
@journal{Herm22,
	title = {Fast and efficient image novelty detection based on mean-shifts},
	journal = {Sensors | Unusual Behavior Detection Based on Machine Learning},
	year = {2022},
	author = {Matthias Hermann and Georg Umlauf and Bastian Goldl{\"u}cke and Matthias O. Franz}
}
```

We organized the main results as IPython Notebooks under ./notebooks for reproducing the results.

# Installation of the package
    
      pip install -f requirements.txt
      python setup.py install
      
# Example

    from meanshift import *

    X, X_valid, X_test = Cifar10_OneClass(train_classes=[9], balance=False, download=True)[0]
    X, X_valid, X_test = asreshape(X, X_valid, X_test, shape=(3, 32, 32))

    model = DeepMeanShift(features="eff")
    model = model.fit(X, verbose=True)
    scores = model.score(X_valid)

    model.evaluate(X_valid, X_test)
    #>> 0.8485..
      
      
      
      


