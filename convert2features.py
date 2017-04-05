# coding: utf-8

import pandas as pd
import numpy as np
import json
import prettytable
import time
import os
from collections import defaultdict, Counter
import feature_config as fconf

class LoadFeatures:
    def __init__(self):
        self.fea_file_addr = "/mnt/work/liubin/recsys2017/features/"
        self.fea_file_dic = json.load(open("/mnt/work/liubin/recsys2017/config/feature_file_dict.json"))
        self.vec_fea_dic = json.load(open("/mnt/work/liubin/recsys2017/config/vec_fea_dict.json"))
        self.uidlist = json.load(open("/mnt/work/liubin/recsys2017/features/user_ids.json"))
        self.iidlist = json.load(open("/mnt/work/liubin/recsys2017/features/item_ids.json"))
        self.f_datadic = {}
        
    def show_features(self, return_list=True):
        num_features = set(self.fea_file_dic.keys())
        num_features = num_features - set(self.vec_fea_dic.keys())

        pair_features = set(json.load(open("/mnt/work/liubin/recsys2017/features/pair_features.json")))

        all_features = num_features | pair_features
        # fea_names = prettytable.PrettyTable()
        # fea_names.add_column("feature_name", sorted(all_features))
        # fea_names.align["feature_name"] = "l" 
        # print fea_names
        if return_list:
            return sorted(all_features)

    def add_feature_files(self, new_feature_file, ftype="num"):
        print "---------- old feature files ----------\n"
        feafiles = json.load(open("/mnt/work/liubin/recsys2017/config/feature_files.json"))
        print feafiles

        if ftype == "num":
            feafiles["num_features"].append(new_feature_file)
        elif ftype == "vec":
            feafiles["vec_features"].append(new_feature_file)

        print "---------- new feature files ----------\n"
        print feafiles
        json.dump(feafiles, open("/mnt/work/liubin/recsys2017/config/feature_files.json", "w"))
        
    def reload_config(self):
        self.fea_file_dic = json.load(open("/mnt/work/liubin/recsys2017/config/feature_file_dict.json"))
        self.show_features(return_list=False)
        
    def load_features(self, flist, key="u", idlist=None):
        """
            key: "u" or "i"
            flist: [features1, features2,,,,] must match key!
        """
        # load files
        ffdic = defaultdict(list)
        for tfea in flist:
            if tfea in self.vec_fea_dic:
                ffdic[self.fea_file_dic[tfea]].extend(self.vec_fea_dic[tfea])
            else:
                ffdic[self.fea_file_dic[tfea]].append(tfea)
        
        for tf in ffdic.keys():
            if tf in self.f_datadic:
                print "already loaded: ", tf
                continue
            print "loading", tf, "..." 
            self.f_datadic[tf] = pd.read_csv(tf, sep="\t")
        print "loaded all feature files!"
        
        # config
        if key == "u":
            id_col = "u_id"
            if not idlist:
                idlist = self.uidlist
        elif key == "i":
            id_col = "i_id"
            if not idlist:
                idlist = self.iidlist
        
        # combine features
        df_ans = pd.DataFrame()
        df_ans[id_col] = idlist
        
        for tff in ffdic:
            tdf = self.f_datadic[tff][[id_col] + ffdic[tff]]
            df_ans = df_ans.merge(tdf, on=id_col, how="left")
        
        return df_ans
    
    def add_newfeatures_from_pandas(self, df_feas, fname, ftype="num"):

        df_feas.to_csv(self.fea_file_addr + fname, index=None, sep="\t")    

        fea_files_addr = "/mnt/work/liubin/recsys2017/config/feature_files.json"

        with open(fea_files_addr) as inf:
            fea_files = json.load(inf)

        if ftype == "num":
            fea_files['num_features'].append(self.fea_file_addr + fname)
        elif ftype == "vec":
            fea_files['vec_features'].append(self.fea_file_addr + fname)

        json.dump(fea_files, open(fea_files_addr, 'w'), indent=True)
        os.system("python /mnt/work/liubin/recsys2017/config/BuildFeatureFileDict.py")
        
        self.reload_config()
        return "saved feature file"
    

class PairFeatures:
    
    def __init__(self):
        self.attribute_match_rules = [("ori_i_country", "ori_u_country", "country"), 
                                      ("ori_i_discipline_id", "ori_u_discipline_id", "discipline_id"), 
                                      ("i_career_level", "u_career_level", "career_level"), 
                                      ("ori_i_industry_id", "ori_u_industry_id", "industry_id"),
                                      ("ori_i_region", "ori_u_region", "region")]
        self.need_features = []        
        self.fea_dic = {
            "p_title_jobroles_match_num": (["ori_u_jobroles"], ['ori_i_title'], lambda tdf: self.__cols_match_num(tdf, "ori_u_jobroles", "ori_i_title", "p_title_jobroles_match_num")),
            "p_title_jobroles_jaccard_sim": (["ori_u_jobroles"], ["ori_i_title"], lambda tdf: self.__cols_match_jsim(tdf, "ori_u_jobroles", "ori_i_title", "p_title_jobroles_jaccard_sim")),
            "p_tags_jobroles_match_num": (["ori_u_jobroles"], ["ori_i_tags"], lambda tdf: self.__cols_match_num(tdf, "ori_u_jobroles", "ori_i_tags", "p_tags_jobroles_match_num")),
            "p_tags_jobroles_jaccard_sim": (["ori_u_jobroles"], ['ori_i_tags'], lambda tdf: self.__cols_match_jsim(tdf, "ori_u_jobroles", "ori_i_tags", "p_tags_jobroles_jaccard_sim")),
            "p_title_poshistitle_match_num": (["u_his_pos_ititles"], ["ori_i_title"], lambda tdf: self.__cols_match_num(tdf, "u_his_pos_ititles", "ori_i_title", "p_title_poshistitle_match_num")),
            "p_title_neghistitle_match_num": (["u_his_neg_ititles"], ["ori_i_title"], lambda tdf: self.__cols_match_num(tdf, "u_his_neg_ititles", "ori_i_title", "p_title_neghistitle_match_num")),
            "p_tag_poshistag_match_num": (["u_his_pos_itags"], ["ori_i_tags"], lambda tdf: self.__cols_match_num(tdf, "u_his_pos_itags", "ori_i_tags", "p_tag_poshistag_match_num")),
            "p_tag_neghistag_match_num": (["u_his_neg_itags"], ["ori_i_tags"], lambda tdf: self.__cols_match_num(tdf, "u_his_neg_itags", "ori_i_tags", "p_tag_neghistag_match_num")),
            
            "p_attribute_match_num": ([t[1] for t in self.attribute_match_rules], [t[0] for t in self.attribute_match_rules], self.__attribute_match_num),
            
            "p_country_match": (["ori_u_country"], ["ori_i_country"], self.__attribute_match),
            "p_discipline_id_match": (["ori_u_discipline_id"], ["ori_i_discipline_id"], self.__attribute_match),
            "p_career_level_match": (["u_career_level"], ["i_career_level"], self.__attribute_match),
            "p_industry_id_match": (["ori_u_industry_id"], ["ori_i_industry_id"], self.__attribute_match),
            "p_region_match": (["ori_u_region"], ["ori_i_region"], self.__attribute_match),
            
            "p_career_level_gap": (["u_career_level"], ["i_career_level"], self.__career_level_gap),
            # "p_region_adjoin": (["ori_u_region"], ["ori_i_region"])

            "p_country_hisratio": (["u_his_ctyratio"], ['ori_i_country'], lambda tdf: self.__cal_hisratio(tdf, "u_his_ctyratio", "ori_i_country", "p_country_hisratio")),
            "p_disciplineid_hisratio": (["u_his_dplratio"], ['ori_i_discipline_id'], lambda tdf: self.__cal_hisratio(tdf, "u_his_dplratio", "ori_i_discipline_id", "p_disciplineid_hisratio")),
            "p_industryid_hisratio": (["u_his_idtyratio"], ['ori_i_industry_id'], lambda tdf: self.__cal_hisratio(tdf, "u_his_idtyratio", "ori_i_industry_id", "p_industryid_hisratio")),
            "p_region_hisratio": (["u_his_regionratio"], ['ori_i_region'], lambda tdf: self.__cal_hisratio(tdf, "u_his_regionratio", "ori_i_region", "p_region_hisratio")),
            "p_careerlevel_hisratio": (["u_his_clratio"], ['i_career_level'], lambda tdf: self.__cal_hisratio(tdf, "u_his_clratio", "i_career_level", "p_careerlevel_hisratio"))
        }
        self.cookie = {}
    
    def get_feature_needed(self, fealist):
        
        ufeaset, ifeaset = [], []
        for tfea in fealist:
            if tfea in self.fea_dic:
                ufeaset.extend(self.fea_dic[tfea][0])
                ifeaset.extend(self.fea_dic[tfea][1])
        ufeaset, ifeaset = set(ufeaset), set(ifeaset)
        return ufeaset, ifeaset
    
    def generate_features_inplace(self, df_pair, fealist):
        funs = set([self.fea_dic[t][2] for t in fealist])
        for tfun in funs:
            tfun(df_pair)
    
    def __str_match(self, tstr1, tstr2, mode="num"):
        if type(tstr1) == float or type(tstr2) == float or tstr1 == "-1" or tstr2 == "-1":
            return 0.0

        tset1, tset2 = set(tstr1.split(",")), set(tstr2.split(","))
        if mode == "num":
            return float(len(tset1 & tset2))
        elif mode == "jsim":
            return float(len(tset1 & tset2)) / len(tset1 | tset2)

    def __attribute_match(self, df_pair):
        for tcol1, tcol2, colname in self.attribute_match_rules:
            df_pair["p_" + colname + "_match"] = np.array(df_pair[tcol1] == df_pair[tcol2], dtype=np.int32)
        
    def __career_level_gap(self, df_pair):
        df_pair['p_career_level_gap'] = df_pair['i_career_level'] - df_pair['u_career_level']
    
    def __attribute_match_num(self, df_pair):
        match_num = np.zeros(len(df_pair))
        for tcol1, tcol2, _ in self.attribute_match_rules:
            match_num += np.array(df_pair[tcol1] == df_pair[tcol2])
        df_pair['p_attribute_match_num'] = match_num
        
    def __cols_match_num(self, df_pair, col_a, col_b, out_col):
        df_pair[out_col] = [self.__str_match(tc1, tc2, mode="num") for tc1, tc2 in zip(df_pair[col_a], df_pair[col_b])]

    def __cols_match_jsim(self, df_pair, col_a, col_b, out_col):
        df_pair[out_col] = [self.__str_match(tc1, tc2, mode="jsim") for tc1, tc2 in zip(df_pair[col_a], df_pair[col_b])]

    def __cal_hisratio(self, df_pair, col_vec, icol, out_col):
        tarray = np.zeros(shape=(len(df_pair), ))
        if icol in self.cookie:
            keyset = self.cookie[icol]
        else:
            keyset = set(df_pair[icol])

        for tkey in keyset:
            ids = np.array(df_pair[icol] == tkey)

            tkey = "".join(str(tkey).split("_"))
            tcol = col_vec + "_" + tkey
            tarray[ids] = df_pair[tcol][ids]

        df_pair[out_col] = tarray


class Convert2feature:
    
    def __init__(self, features_dic=None, feature_list=None):
        
        if not features_dic:
            self.ufeatures = fconf.user_features
            self.ifeatures = fconf.item_features
            self.pfeatures = fconf.pair_features
        else:
            self.ufeatures = features_dic['user_features']
            self.ifeatures = features_dic['item_features']
            self.pfeatures = features_dic['pair_features']

        if not feature_list:
            self.features_names = self.ufeatures + self.ifeatures + self.pfeatures
        else:
            self.ufeatures = []
            self.ifeatures = []
            self.pfeatures = []
            for tf in feature_list:
                if tf[0] == "u" or tf[:2] == "lb":
                    self.ufeatures.append(tf)
                elif tf[0] == "i":
                    self.ifeatures.append(tf)
                elif tf[0] == "p":
                    self.pfeatures.append(tf)
                else:
                    print "Error! ", tf
            print "init with %d user features, %d item features, %d pair features, Total: %d" %(len(self.ufeatures), len(self.ifeatures), len(self.pfeatures), len(feature_list))
            self.features_names = feature_list

        df_target_user = pd.read_csv("/mnt/work/haobin/recsys2017/dataset/origindataset/targetUsers.csv", sep="\t")
        self.target_users = list(df_target_user.iloc[:,0])
                    
        # find feature needed
        self.pf = PairFeatures()
        uf_need, if_need = self.pf.get_feature_needed(self.pfeatures)
        self.ufeatures = list(set(self.ufeatures) | uf_need | set(['u_premium']))
        self.ifeatures = list(set(self.ifeatures) | if_need | set(['i_is_payed']))

        print "User Features Loading: ", self.ufeatures
        print "Item Features Loading: ", self.ifeatures
        # load features
        loadf = LoadFeatures()        
        self.df_u = loadf.load_features(self.ufeatures, key="u")
        self.df_i = loadf.load_features(self.ifeatures, key="i")
        
        # init df_ans
        self.df_ans = None
        
    def get_features_name(self):
        return self.features_names

    def convert_by_itemid(self, itemid, target_users=None, mode="numpy"):
        # user_features
        
        if not target_users:
            target_users = self.target_users

        self.df_ans = pd.DataFrame()
        self.df_ans["u_id"] = target_users
        self.df_ans = self.df_ans.merge(self.df_u, how="left", on="u_id", copy=False)
        
        # item_features
        df_ifeature = self.df_i[self.df_i['i_id'] == itemid]
        for tif in df_ifeature.columns:
            self.df_ans[tif] = [df_ifeature[tif].iloc[0]] * len(target_users)
        
        # pair_features
        self.pf.generate_features_inplace(self.df_ans, self.pfeatures)
        
        self.df_ans.fillna(-1.0, inplace=True)
        if mode == "numpy":
            return self.df_ans[['u_id', 'i_id'] + self.features_names].values
        elif mode == "pandas":
            return self.df_ans[['u_id', 'i_id'] + self.features_names]
        
    def convert_by_pair(self, df_pair, include_id=True, mode="numpy"):
        base_col = list(df_pair.columns)
        
        btime = time.time()
        print "converting user features...", 
        df_pair = df_pair.merge(self.df_u, on="u_id", how="left", copy=False)
        print "cost: ", time.time() - btime
        
        btime = time.time()        
        print "converting item features...",
        df_pair = df_pair.merge(self.df_i, on="i_id", how="left", copy=False)
        print "cost: ", time.time() - btime
        
        btime = time.time()        
        print "converting pair features...",
        self.pf.generate_features_inplace(df_pair, self.pfeatures)
        print "cost: ", time.time() - btime

        print "filling nan..."
        df_pair.fillna(-1.0, inplace=True)
        
        if include_id:
            out_cols = ['u_id', 'i_id'] + self.features_names
        else:
            out_cols = self.features_names

        if mode == "numpy":
            ans_array = df_pair[out_cols].values
            print "shape: ", ans_array.shape 
            return ans_array, self.features_names
        elif mode == "pandas":
            return df_pair[base_col + self.features_names], self.features_names


    def generate_labels(self, df_pair, mode="numpy", columns=[u'u_premium', u'i_is_payed', u'impression_cnt', u'click_cnt', u'bookmarked_cnt', u'reply_cnt', u'delete_cnt', u'recruiter_cnt']): 

        if "u_premium" in columns:
            df_pair = df_pair.merge(self.df_u[['u_id', 'u_premium']], on="u_id", how="left", copy=False)

        if "i_is_payed" in columns:
            df_pair = df_pair.merge(self.df_i[['i_id', "i_is_payed"]], on="i_id", how="left", copy=False)

        if mode == "numpy":
            return df_pair[columns].values
        elif mode == "pandas":
            return df_pair[columns]
