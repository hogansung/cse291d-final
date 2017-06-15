#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

// const parameters 
const int K = 10;
const double lambda = 20;
const double mf_p = 1;
const int MAX_ITER = 100;
const double eps = 1e-4;
const double MONTH_SCALE = 1.0;

// global-wise parameters
int NU;
int NI;
const int NM = 12;
int NTN;
int NTT;

struct Tuple {
    int user;
    int item;
    int rate;
    int month;
    Tuple() {}
    Tuple(int u, int i, int r, int m): user(u), item(i), rate(r), month(m) {}
};

default_random_engine generator;
uniform_real_distribution<double> distribution(0.0,1.0/sqrt(K));

unordered_map<string, int> map_user_idx;
unordered_map<int, string> map_idx_user;
unordered_map<string, int> map_item_idx;
unordered_map<int, string> map_idx_item;

vector<Tuple> tn_tuple;
vector<Tuple> tt_tuple;

char in[100010];
//string suffix = "";
string suffix = "_60";

const int TARGET = 0;
unordered_set<int> targetItem;

// Read Mappings
void readMappings() {
    { // user part
        string userPath = string("../tab/userList") + suffix;
        FILE* pfile = fopen(userPath.c_str(), "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NU);
        for (int i = 0; i < NU; i++) {
            fscanf(pfile, "%s", in);
            map_user_idx[in] = i;
            map_idx_user[i] = in;
        }
        fclose(pfile);
    }
    { // item part
        string itemPath = string("../tab/itemList") + suffix;
        FILE* pfile = fopen(itemPath.c_str(), "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NI);
        for (int i = 0; i < NI; i++) {
            fscanf(pfile, "%s", in);
            map_item_idx[in] = i;
            map_idx_item[i] = in;
        }
        fclose(pfile);
    }
}

// Read Data
void readData() {
    { // train part
        string tnPath = string("../tab/tnList") + suffix;
        FILE* pfile = fopen(tnPath.c_str(), "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NTN);
        int user, item, rate;
        int year, month, date, day;
        for (int i = 0; i < NTN; i++) {
            fscanf(pfile, "%d,%d,%d,%d,%d,%d,%d", &user, &item, &rate, &year, &month, &date, &day);
            if (TARGET == -1 or targetItem.find(item) != targetItem.end()) {
                tn_tuple.emplace_back(user, item, rate, month-1);
            }
        }
        fclose(pfile);
    }
    { // test part
        string ttPath = string("../tab/ttList") + suffix;
        FILE* pfile = fopen(ttPath.c_str(), "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NTT);
        int user, item, rate;
        int year, month, date, day;
        for (int i = 0; i < NTT; i++) {
            fscanf(pfile, "%d,%d,%d,%d,%d,%d,%d", &user, &item, &rate, &year, &month, &date, &day);
            if (TARGET == -1 or targetItem.find(item) != targetItem.end()) {
                tt_tuple.emplace_back(user, item, rate, month-1);
            }
        }
        fclose(pfile);
    }
}

void reMappings() {
    NTN = tn_tuple.size();
    NTT = tt_tuple.size();

    unordered_map<string, int> local_map_user_idx;
    unordered_map<int, string> local_map_idx_user;
    unordered_map<string, int> local_map_item_idx;
    unordered_map<int, string> local_map_idx_item;

    vector<Tuple> local_tn_tuple;
    vector<Tuple> local_tt_tuple;

    NU = 0;
    NI = 0;
    for (auto tuple : tn_tuple) {
        int user = tuple.user;
        int item = tuple.item;
        int rate = tuple.rate;
        int month = tuple.month;
        
        if (local_map_user_idx.find(map_idx_user[user]) == local_map_user_idx.end()) {
            local_map_user_idx[map_idx_user[user]] = NU;
            local_map_idx_user[NU] = map_idx_user[user];
            NU += 1;
        }
        if (local_map_item_idx.find(map_idx_item[item]) == local_map_item_idx.end()) {
            local_map_item_idx[map_idx_item[item]] = NI;
            local_map_idx_item[NI] = map_idx_item[item];
            NI += 1;
        }

        local_tn_tuple.emplace_back(local_map_user_idx[map_idx_user[user]],
                                    local_map_item_idx[map_idx_item[item]],
                                    rate, month);
    }
    for (auto tuple : tt_tuple) {
        int user = tuple.user;
        int item = tuple.item;
        int rate = tuple.rate;
        int month = tuple.month;

        if (local_map_user_idx.find(map_idx_user[user]) == local_map_user_idx.end()) {
            local_map_user_idx[map_idx_user[user]] = NU;
            local_map_idx_user[NU] = map_idx_user[user];
            NU += 1;
        }
        if (local_map_item_idx.find(map_idx_item[item]) == local_map_item_idx.end()) {
            local_map_item_idx[map_idx_item[item]] = NI;
            local_map_idx_item[NI] = map_idx_item[item];
            NI += 1;
        }

        local_tt_tuple.emplace_back(local_map_user_idx[map_idx_user[user]],
                                    local_map_item_idx[map_idx_item[item]],
                                    rate, month);
    }

    map_user_idx = local_map_user_idx;
    map_idx_user = local_map_idx_user;
    map_item_idx = local_map_item_idx;
    map_idx_item = local_map_idx_item;

    tn_tuple = local_tn_tuple;
    tt_tuple = local_tt_tuple;
}

// Read Cluster
void readCluster() {
    FILE* pfile = fopen("../tab/c_cluster.txt", "r");
    assert(pfile != NULL);
    for (int i = 0; i < 11; i++) {
        int nc, c;
        fscanf(pfile, "%d", &nc);

        for (int j = 0; j < nc; j++) {
            fscanf(pfile, "%d", &c);
            if (i == TARGET) {
                targetItem.insert(c);
            }
        }
    }
    fclose(pfile);
}

double calERR(vector<Tuple>& data, vector<double>& pred, int size,
  double A, MatrixXd& BU, MatrixXd& BI, MatrixXd& BM, MatrixXd& P, MatrixXd& Q) {
    double err = 0;
    for (int i = 0; i < size; i++) {
        double diff = (pred[i] - data[i].rate);
        err += diff * diff;
    }
    err += lambda * (BU.squaredNorm() + BI.squaredNorm() + MONTH_SCALE * BM.squaredNorm()
      + mf_p * (P.squaredNorm() + Q.squaredNorm()));
    return err;
}

double calMSE(vector<Tuple>& data, vector<double>& pred, int size) {
    double err = 0;
    for (int i = 0; i < size; i++) {
        double diff = (pred[i] - data[i].rate);
        err += diff * diff;
    }
    return err / data.size();
}

void predict(vector<Tuple>& data, vector<double>& pred, int size,
  double A, MatrixXd& BU, MatrixXd& BI, MatrixXd& BM, MatrixXd& P, MatrixXd& Q) {
    for (int i = 0; i < size; i++) {
        int user = data[i].user;
        int item = data[i].item;
        int month = data[i].month;
        pred[i] = A + BU(user,0) + BI(item,0) + MONTH_SCALE * BM(month,0) 
          + mf_p * P.row(user) * Q.row(item).transpose();
    }
}

void setRandom(MatrixXd& M, int nr, int nc) {
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            M(i, j) = distribution(generator);
        }
    }
}

void saveModel(double A, MatrixXd& BU, MatrixXd& BI, MatrixXd& BM, MatrixXd& P, MatrixXd& Q) {
    /* Meta Information
     * NU NI 
     * A (1, 1)
     * BU (NU, 1)
     * BI (NI, 1)
     * BM (NM, 1) */
    {
        FILE* pfile = fopen("../mdl/meta.mat", "w");
        assert(pfile != NULL);
        fprintf(pfile, "%d %d\n", NU, NI);
        fprintf(pfile, "%f\n", A);
        for (int i = 0; i < NU; i++) {
            fprintf(pfile, "%f%c", BU(i,0), i == NU-1 ?'\n' :' ');
        }
        for (int i = 0; i < NI; i++) {
            fprintf(pfile, "%f%c", BI(i,0), i == NI-1 ?'\n' :' ');
        }
        for (int i = 0; i < NM; i++) {
            fprintf(pfile, "%f%c", BM(i,0), i == NM-1 ?'\n' :' ');
        }
    }

    /* User Matrix: 
     * NU K 
     * P (NU, K) */
    {
        FILE* pfile = fopen("../mdl/user.mat", "w");
        assert(pfile != NULL);
        fprintf(pfile, "%d %d\n", NU, K);
        for (int i = 0; i < NU; i++) {
            for (int j = 0; j < K; j++) {
                fprintf(pfile, "%f%c", P(i, j), j == K-1 ?'\n' :' ');
            }
        }
        fclose(pfile);
    }

    /* Item Matrix: 
     * NI K 
     * Q (NI, K) */
    {
        FILE* pfile = fopen("../mdl/item.mat", "w");
        assert(pfile != NULL);
        fprintf(pfile, "%d %d\n", NI, K);
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < K; j++) {
                fprintf(pfile, "%f%c", Q(i, j), j == K-1 ?'\n' :' ');
            }
        }
        fclose(pfile);
    }

}

int main() {
    srand(514);

    readMappings();
    if (TARGET != -1) {
        readCluster(); 
    }
    readData();

    if (TARGET != -1) {
        reMappings();
    }

    double A = 0;
    MatrixXd BU = MatrixXd::Zero(NU, 1);
    MatrixXd BI = MatrixXd::Zero(NI, 1);
    MatrixXd BM = MatrixXd::Zero(NM, 1);
    MatrixXd P = MatrixXd::Zero(NU, K);
    setRandom(P, NU, K);
    MatrixXd Q = MatrixXd::Zero(NI, K);
    setRandom(Q, NI, K);

    vector<double> tn_p(NTN, 0);
    vector<double> tt_p(NTT, 0);
    predict(tn_tuple, tn_p, NTN, A, BU, BI, BM, P, Q);
    predict(tt_tuple, tt_p, NTT, A, BU, BI, BM, P, Q);

    double last_tn_err = calERR(tn_tuple, tn_p, NTN, A, BU, BI, BM, P, Q);
    for (int time = 0; time < MAX_ITER; time++) {
        double nA = 0;
        double cA = 0;
        MatrixXd nBU = MatrixXd::Zero(NU, 1);
        MatrixXd nBI = MatrixXd::Zero(NI, 1);
        MatrixXd nBM = MatrixXd::Zero(NM, 1);
        VectorXd cU = VectorXd::Zero(NU);
        VectorXd cI = VectorXd::Zero(NI);
        VectorXd cM = VectorXd::Zero(NM);
        MatrixXd nP = MatrixXd::Zero(NU, K);
        MatrixXd nQ = MatrixXd::Zero(NI, K);
        VectorXd cP = VectorXd::Zero(NU);
        VectorXd cQ = VectorXd::Zero(NI);
        
        for (auto tuple : tn_tuple) {
            int user = tuple.user;
            int item = tuple.item;
            double rate = tuple.rate;
            int month = tuple.month;
            double p = A + BU(user,0) + BI(item,0) + MONTH_SCALE * BM(month,0) 
              + mf_p * P.row(user) * Q.row(item).transpose();
            double diff = (p - rate);

            nA += -diff + A;
            cA += 1;
            nBU(user,0) += -diff + BU(user,0);
            cU(user) += 1;
            nBI(item,0) += -diff + BI(item,0);
            cI(item) += 1;
            nBM(month,0) += -diff + BM(month,0);
            cM(month) += 1;
            nP.row(user) += (-diff + P.row(user) * Q.row(item).transpose()) * Q.row(item);
            cP(user) += Q.row(item).squaredNorm();
            nQ.row(item) += (-diff + P.row(user) * Q.row(item).transpose()) * P.row(user);
            cQ(item) += P.row(user).squaredNorm();
        }

        if (time % 6 == 0) { 
            A = nA / cA;
        } else if (time % 6 == 1) {
            BU = nBU.array() / (cU.array() + lambda);
        } else if (time % 6 == 2) {
            BI = nBI.array() / (cI.array() + lambda);
        } else if (time % 6 == 3) {
            BM = nBM.array() / (cM.array() + lambda);
        } else if (time % 6 == 4) {
            P = nP.array().colwise() / (cP.array() + lambda);
        } else {
            Q = nQ.array().colwise() / (cQ.array() + lambda);
        }

        if (time % 6 == 0) {
            predict(tn_tuple, tn_p, NTN, A, BU, BI, BM, P, Q);
            predict(tt_tuple, tt_p, NTT, A, BU, BI, BM, P, Q);

            double tn_mse = calMSE(tn_tuple, tn_p, NTN);
            double tt_mse = calMSE(tt_tuple, tt_p, NTT);
            double tn_err = calERR(tn_tuple, tn_p, NTN, A, BU, BI, BM, P, Q);
            double tt_err = calERR(tt_tuple, tt_p, NTT, A, BU, BI, BM, P, Q);
            printf("%d %f %f %f %f\n", time, tn_mse, tt_mse, tn_err, tt_err);

            if (abs(last_tn_err - tn_err) / tn_err < eps) {
                break;
            } else {
                last_tn_err = tn_err;
            }
        }
    }

    /* Save model */
    saveModel(A, BU, BI, BM, P, Q);
}
