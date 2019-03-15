
#Apriori algo
class Apriori:
    def __init__(self,X,minsup=0,minconf=0):
        self.min_sup=minsup*X.shape[0]
        self.min_conf=minconf
        self.df=X
        self.itemset={'1':{}}
        self.rule=[]
        self.X_ = np.array(self.df)
        self.strong_rule=[]
    def generate(self):
        """generate length 1 frequent itemset"""
        for col in self.df.columns:
            temp_dict=self.df[col].value_counts()
            temp_dict.index=[tuple([x]) for x in temp_dict.index]
            temp_dict=temp_dict.to_dict()
            for item in list(temp_dict.keys()):
                if temp_dict[item]<self.min_sup:
                    temp_dict.pop(item)
            self.itemset['1']={**self.itemset['1'],**temp_dict}
            self.rule.extend([x for x in list(temp_dict.keys())])
        length = self.df.shape[1]
        self.scan_generate(rules=self.rule,iteration=length,curlevel=1)
        return self.itemset
    def scan_generate(self,iteration,rules,curlevel=1):
        """generate frequent itemset length by length
        for the length k, only consider the subset with the first k-1 items in same"""
        if len(rules)<=1:
            return
        self.itemset[str(curlevel+1)]={}
        temp_rules=[]
        if iteration==0:
            return
        iteration-=1
        for i in range(len(rules)-1):
            for j in range(i+1,len(rules)):
                if set(rules[i]).intersection(set(rules[j])).__len__()!=curlevel-1:
                    continue
                target_set=set(sorted(set(rules[i]).union(set(rules[j]))))
                temp_sup=self.get_count(self.X_,target_set)
                if temp_sup>self.min_sup and tuple(target_set) not in self.itemset[str(curlevel+1)].keys():
                    self.itemset[str(curlevel + 1)][tuple(target_set)]=temp_sup
                    temp_rules.append(tuple(target_set))

        self.scan_generate(rules=temp_rules,iteration=iteration,curlevel=curlevel+1)
        return

    def get_count(self,X,target_set):
        """iterate the data set to find the support for the itemset"""
        count=0
        # print(target_set)
        for i in range(X.shape[0]):

            if target_set.issubset(set(X[i])):
                count+=1
            else:
                pass
        return count
    def generate_rules(self,cur_key):
        """iteratively generate all subset for the current itemset and generate the rule
        with high confidence"""
        if int(cur_key)<=1:
            return
        last_key=str(int(cur_key)-1)
        whole_sets=list(self.itemset[cur_key].keys())
        if not whole_sets:
            self.generate_rules(last_key)
        while whole_sets:
            whole_set_key=whole_sets.pop()
            whole_set=set(whole_set_key)
            # print(whole_set)
            for i in range(1, int(cur_key)):
                sub_set = list(self.itemset[str(i)].keys())
                for item_set in sub_set:
                    if not set(item_set).issubset(whole_set):
                        continue
                    complementary_set =whole_set.difference(set(item_set))
                    complementary_set_len=complementary_set.__len__()
                    # print(complementary_set,set(item_set))
                    if tuple(complementary_set) in self.itemset[str(complementary_set_len)].keys():
                        if self.itemset[cur_key][whole_set_key]/self.itemset[str(i)][item_set]>self.min_conf:
                            self.strong_rule.append([item_set,complementary_set])
        self.generate_rules(last_key)
    def display_rule(self):
        for item in self.strong_rule:
            print(item[0],'----->',item[1])



if __name__ == '__main__':
    import pandas as pd
    import os
    import numpy as np
    from functools import reduce
    # warnings.filterwarnings(category=SettingWithCopyWarning,action='ignore')
    os.chdir('c:/users/LZC/desktop/data mining')
    col_name = ['age', 'workclass', 'fnlweight', 'education', 'education-num', 'martial-status',
                'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
    df = pd.read_csv('adult.data.txt', names=col_name)
    df['age'] = np.digitize(df['age'], bins=[0, 18, 30, 50, 100])
    df['fnlweight'] = pd.qcut(df['fnlweight'], q=[0, 0.25, 0.5, 0.75, 1], labels=False)
    df['hours-per-week'] = np.digitize(df['hours-per-week'],bins=np.arange(0,np.max(df['hours-per-week']),10))
    df.loc[np.nonzero(df['capital-gain']!=0)[0],'capital-gain'] = pd.qcut(df['capital-gain'][np.nonzero(df['capital-gain']!=0)[0]] , q=[0, 0.25, 0.5, 0.75, 1], labels=False).copy()
    df.loc[np.nonzero(df['capital-loss']!=0)[0],'capital-loss'] = pd.qcut(df['capital-loss'][np.nonzero(df['capital-loss']!=0)[0]] , q=[0, 0.25, 0.5, 0.75, 1], labels=False).copy()
    for i in range(len(df.columns)):
        df[df.columns[i]]=df[df.columns[i]].apply(lambda x:df.columns[i]+'.'+ str(x))
    Apriori=Apriori(minsup=0.8,X=df,minconf=0.9)
    Apriori.generate()
    Apriori.generate_rules(list(Apriori.itemset.keys())[-1])
    # Apriori.strong_rule
    # Apriori.display_rule()
    print(Apriori.itemset)