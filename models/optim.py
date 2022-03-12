import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


class ThreshholdOptimizer:
    def __init__(self, r, p):
        self.r = r
        self.p = p
        self.theta = 0.5

    def fit(self, y_true, y_pred_proba):
        self.thresh = np.linspace(0, 1, 1000)
        self.recall = np.zeros(len(self.thresh))
        self.risk = np.zeros_like(self.recall)
        tps = np.zeros_like(self.recall)
        self.specificity = np.zeros_like(self.recall)
        self.precision = np.zeros_like(self.recall)
        j = 0
        for tr in self.thresh:
            pred = np.where(y_pred_proba>=tr, 1, 0).reshape(-1, 1)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred).ravel()
            tps[j] = tp
            self.risk[j] = fp*(self.p/self.r) + fn*(self.r/self.p)
            self.recall[j] = tp / (tp + fn)
            self.specificity[j] = tn / (tn + fp)
            self.precision[j] = tp / (tp + fp)
            j += 1
        self.min_pos = np.argmin(self.risk)
        self.theta = self.thresh[self.min_pos]

    def plot_threshold(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        auc = str(round(metrics.auc(self.specificity, self.recall), 2))
        opt_prec, opt_rec = self.precision[self.min_pos], self.recall[self.min_pos]
        plt.suptitle('AUC='+auc+' rec=' +str(round(opt_rec*100,2))+' prec='+str(round(opt_prec*100, 2)))
        ax1.plot(1-self.specificity, self.recall, linewidth=2)
        ax1.set_xlabel('False positive rate [FP/N]')
        ax1.set_ylabel('Recall [TP/P]')
        ax1.set_title('ROC Curve')
        ax2.plot(self.recall, self.precision, linewidth=2)
        ax2.set_xlabel('Recall [TP/P = TP/(TP+FN)]')
        ax2.set_ylabel('Precision [TP/(TP+FP)]')
        ax2.axvline(x=opt_rec, linestyle='dashed', color='red', label="optimized threshhold")
        ax2.axhline(y=opt_prec, linestyle='dashed', color='red')
        ax2.axvline(x=self.recall[501], linestyle='dashed', color='black', label="threshhold")
        ax2.axhline(y=self.precision[501], linestyle='dashed', color='black')
        ax2.set_title('Recall-Precision Curve')
        ax3.plot(self.thresh, self.risk, linewidth=2)
        ax3.set_title('Risk Minimization')
        ax3.set_xlim(-0.1, 1.1)
        try:
            ymax = max(self.risk[self.min_pos-100], self.risk[self.min_pos+100])
            ax3.set_ylim(self.risk[self.min_pos]-20, ymax+100)
        except:
            pass
        ax3.axvline(x=self.theta, linestyle='dashed', color='red', label="optimized threshhold")
        ax3.axvline(x=0.5, linestyle='dashed', color='black', label="threshhold")
        ax3.set_xlabel('Treshhold')
        ax3.set_ylabel('Risk')
        ax2.legend()
        ax3.legend()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        return fig

    def predict(self, y_pred_proba):
        y_pred = np.where(y_pred_proba >= self.theta, 1, 0)
        return y_pred

    def confusion_matrix(self, y_pred_proba, y_true):
        y_pred = self.predict(y_pred_proba)
        return metrics.confusion_matrix(y_true, y_pred)
