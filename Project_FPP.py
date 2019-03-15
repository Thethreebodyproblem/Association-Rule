class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        """create the definition of the tree"""
        """
        nodelink: refer to the samiliar node
        childern: children node 
        parent:   parent node
        """
        self.name=nameValue
        self.count=numOccur
        self.nodelink=None
        self.parent=parentNode
        self.children={}

    def inc(self,numoccur):
        self.count+=numoccur
class FP_Growth:
    def __init__(self,X,minsup=0,minconf=0):
        """create the FP tree"""
        """
        header_linklist: use to create the linkedlist between the node
        header_linklist_: header node for the linkedlist
        X_:   array of data frame
        itemset: total length 1 frequent item
        """
        self.min_sup=minsup*X.shape[0]
        self.min_conf=minconf
        self.df=X
        self.itemset={}
        self.header={}
        self.header_linklist={}
        self.header_linklist_={}
        self.rule=[]
        self.X_ = np.array(self.df)
        self.root=treeNode(nameValue='root',numOccur=0,parentNode=None)
    def generate_header(self):
        """create the length 1 frequent itemset"""
        for col in self.df.columns:
            temp_dict=self.df[col].value_counts().to_dict()
            for item in list(temp_dict.keys()):
                if temp_dict[item]<self.min_sup:
                    temp_dict.pop(item)
            self.itemset={**self.itemset,**temp_dict}

        self.header={k:v for k,v in sorted(self.itemset.items(),key=lambda x:x[1],reverse=True)}
        return self.header
    def scan_build_tree(self):
        """iterate over the whole data set and create the FP tree"""
        for line in self.X_:
            root=self.root
            target_set=set(line).intersection(set(self.header.keys()))
            if not target_set:
                continue
            target_set_sort=sorted(target_set,key=lambda x:self.header[x],reverse=True)
            for item in target_set_sort:
                next_root=self.insert_update_tree(root,item)
                root=next_root
    def insert_update_tree(self,root,node):
        """update and insert the node to the current FP tree
        create the linkedlist between the node with the same name"""
        if node not in root.children.keys():
            treenode=treeNode(node,1,root)
            if node not in self.header_linklist.keys():
                self.header_linklist[node]=treenode
                self.header_linklist_[node] = treenode
            else:
                self.header_linklist[node].nodelink=treenode
                self.header_linklist[node]=self.header_linklist[node].nodelink
            root.children[node]=treenode
        else:
            root.children[node].inc(1)
        return root.children[node]
    def dis_dfs(self,root,inc=1):
        """Use depth first search to print out the tree"""
        print('    ' * inc, root.name + ' ' + str(root.count))
        if root.children:
            for item in root.children.keys():
                self.dis_dfs(root.children[item],inc+1)
    def generate_itemset(self,path,cur_node,last_count):
        if cur_node.children:
            for item in cur_node.children.keys():
                if cur_node.children[item].count>self.min_sup:
                    self.generate_itemset(path+[item],cur_node.children[item],cur_node.children[item].count)
                else:
                    if len(path)>1:
                        self.itemset[tuple(path)]=last_count
                    return
        return

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
    CLF=FP_Growth(minsup=0.7,X=df,minconf=0.7)
    CLF.generate_header()
    CLF.scan_build_tree()
    CLF.dis_dfs(CLF.root)

