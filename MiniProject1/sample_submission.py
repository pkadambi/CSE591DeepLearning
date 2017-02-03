'''
Created on Feb 2, 2017

@author: Prad
'''
#sample_submission.py
import numpy as np
import time
class regressor(object):
    """
    This is a sample class for miniproject 1.
    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D. 
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data):
        self.x, self.y = data        
        # Here is where your training and all the other magic should happen. 
        # Once trained you should have these parameters with ready. 
#         print self.x.shape
#         print self.y.shape

        #add the row of 1's so that we can learn the b value (w0)
        self.x = np.hstack([np.ones([self.x.shape[0],1]),self.x])
        #as a result of the previous line now we also have the correct number of elements in W
        self.w = np.random.rand(self.x.shape[1],1)
        
        self.eta = 0.0001
        
        
        
        self.alpha = 0.001
#         print 'NEW RUN****************************************'
        self._do_naive_gradient_descent()
#         self._do_gradient_descent_ridge()

        
        
        #dont wwant to do b alone
#         self.b = np.random.rand(1)
        
    
    #unregularized gradient descent
    def _do_naive_gradient_descent(self):
        
        #repeat until we get the RMSE that we want
        ecurr = self._get_squared_error(self.x, self.y, self.w)
        
        eprev =  ecurr+1 #initialize previous error to 1+ current error
        i=0
        while((ecurr-eprev)>0.0001):
            
            if (i%100000==0):
                print i
                print '\nCurrent Error:'
                print ecurr
                print '\nPrev Error:'
                print eprev
                print '\nWeights:'
                print self.w.T
                print '\nCurrent update:'
                
                update = self.eta*self._get_gradient(self.x,self.y,self.w)
                self.w= self.w-update
                
                print update.T
                print '\nUpdated Weights:'
                print self.w.T
                
                print '\nLearning Rate:'
                print self.eta
            
            if(eprev<ecurr):
                self.eta = self.eta/2
#             if(np.abs(eprev-ecurr)/ecurr<0.001):# and np.linalg.norm(update)<3):
#                 self.eta=self.eta/3
            eprev=ecurr
            ecurr = self._get_squared_error(self.x, self.y, self.w)
            
            if (i%100000==0):
                print '\nTraining RMSE: '+ str(np.sqrt(ecurr))
            
            
                print('NEXT ITER--------------------')
            i+=1
#             time.sleep(0.01)
            
        print 'Training RMSE: '+ str(np.sqrt(self._get_squared_error(self.x, self.y, self.w)))
        
        #unregularized gradient descent
    def _do_gradient_descent_ridge(self):
        
        #repeat until we get the RMSE that we want
        ecurr = self._get_squared_error_ridge(self.x, self.y, self.w)
        
        eprev =  ecurr+1 #initialize previous error to 1+ current error
        i=0
        
        rmsecurr=self.rmse(self.y, np.dot(self.x,self.w))
        rmseprev=rmsecurr+1
        while(rmseprev-rmsecurr>0.00001):
            
            if (i%100000==0):
                print i
                print '\nCurrent Error:'
                print ecurr
                print '\nPrev Error:'
                print eprev
                print '\nWeights:'
                print self.w.T
                print '\nCurrent update:'
                
                update = self.eta*self._get_gradient_ridge(self.x,self.y,self.w)
                self.w= self.w-update
                
                print update.T
                print '\nUpdated Weights:'
                print self.w.T
                
                print '\nLearning Rate:'
                print self.eta
            
            if(eprev<ecurr):
                self.eta = self.eta/2
#             if(np.abs(eprev-ecurr)/ecurr<0.001):# and np.linalg.norm(update)<3):
#                 self.eta=self.eta/3
            rmseprev=rmsecurr
            eprev=ecurr
            ecurr = self._get_squared_error_ridge(self.x, self.y, self.w)
            rmsecurr = self.rmse(self.y, np.dot(self.x,self.w))
            
            if (i%100000==0):
                print '\nTraining RMSE: '+ str(np.sqrt(ecurr))
            
            
                print('NEXT ITER--------------------')
            i+=1
#             time.sleep(0.01)
            
        print 'Training RMSE: '+ str(np.sqrt(self._get_squared_error_ridge(self.x, self.y, self.w)))        
    
    
    def rmse (self,  a,  b ): 
    
        return np.sqrt(np.mean((a - b) ** 2))
    def _rmse (self,  data_test,data_targets): 
    
        return np.sqrt(np.mean((data_targets - np.dot(data_targets,self.w)) ** 2))
    
    def _get_gradient(self,x,y,w):
        return  -2*np.dot(x.T,y)+2*np.dot(np.dot(x.T,x),w)
    
    def _get_gradient_ridge(self,x,y,w):
        return  -2*np.dot(x.T,y)+2*np.dot(np.dot(x.T,x),w)+2*self.alpha*self.w
    
    def _get_squared_error_ridge(self,x,y,w):
        # get the L2 norm here (euclidean distance between prediction and actual)
#         manualerr = np.dot(np.transpose((y-np.dot(x,w))),y-(np.dot(x,w)))+self.alpha*np.dot(self.w.T,self.w)
        manualerr = np.sum((y- (np.dot(x,w))) ** 2)/5000
        return np.sqrt(manualerr)
    def _get_squared_error(self,x,y,w):
        # get the L2 norm here (euclidean distance between prediction and actual)
        manualerr = np.dot(np.transpose((y-np.dot(x,w))),y-(np.dot(x,w)))
#         manualerr = np.sum((y- (np.dot(x,w)+self.b)) ** 2)/5000
        
        return manualerr
    def get_params (self):
        """ 
        Method that should return the model parameters.
        Returns:
            tuple of numpy.ndarray: (w, b). 
        Notes:
            This code will return a random numpy array for demonstration purposes.
        """
        
        #REMEMBER  THAT YUOU HAVE TO SEPARATE OUT THE B HERE!!!!!!!!!
        
        return (self.w[1:], self.w[0])

    def get_predictions (self, x):
        #remember that here he's not giving you the row of 1's before the x
        """
        Method should return the outputs given unseen data
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        #add the ones row to make the prediction with the full weights vector
        print x.shape
        x = np.hstack([np.ones([x.shape[0],1]),x])
#         print x
        # Here is where you write a code to evaluate the data and produce predictions.
        return np.dot(x,self.w)

    
if __name__ == '__main__':
    pass 