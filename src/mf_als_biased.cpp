#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;


// const parameters 
const int K = 10;
const double lambda = 5;
const int MAX_ITER = 100;
const double eps = 1e-4;

// global-wise parameters
int NU;
int NI;
int NTN;
int NTT;

struct Tuple {
    int user;
    int item;
    int rate;
    Tuple() {}
    Tuple(int u, int i, int r): user(u), item(i), rate(r) {}
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

// Read Mappings
void readMappings() {
    { // user part
        FILE* pfile = fopen("../tab/userList", "r");
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
        FILE* pfile = fopen("../tab/itemList", "r");
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
        FILE* pfile = fopen("../tab/tnList", "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NTN);
        int user, item, rate;
        int year, month, date, day;
        for (int i = 0; i < NTN; i++) {
            fscanf(pfile, "%d,%d,%d,%d,%d,%d,%d", &user, &item, &rate, &year, &month, &date, &day);
            tn_tuple.emplace_back(user, item, rate);
        }
        fclose(pfile);
    }
    { // test part
        FILE* pfile = fopen("../tab/ttList", "r");
        assert(pfile != NULL);
        fscanf(pfile, "%d", &NTT);
        int user, item, rate;
        int year, month, date, day;
        for (int i = 0; i < NTT; i++) {
            fscanf(pfile, "%d,%d,%d,%d,%d,%d,%d", &user, &item, &rate, &year, &month, &date, &day);
            tt_tuple.emplace_back(user, item, rate);
        }
        fclose(pfile);
    }
}

double calERR(vector<Tuple>& data, vector<double>& pred, int size,
  double A, MatrixXd& BU, MatrixXd& BI) {
    double err = 0;
    for (int i = 0; i < size; i++) {
        double diff = (pred[i] - data[i].rate);
        err += diff * diff;
    }
    err += lambda * (BU.squaredNorm() + BI.squaredNorm());
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
  double A, MatrixXd& BU, MatrixXd& BI) {
    for (int i = 0; i < size; i++) {
        int user = data[i].user;
        int item = data[i].item;
        pred[i] = A + BU(user,0) + BI(item,0);
    }
}

void setRandom(MatrixXd& M, int nr, int nc) {
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            M(i, j) = distribution(generator);
        }
    }
}

void saveModel(double A, MatrixXd& BU, MatrixXd& BI) {
    /* Meta Information
     * NU NI 
     * A (1, 1)
     * BU (NU, 1)
     * BI (NI, 1) */
    {
        FILE* pfile = fopen("../mdl/meta.mat", "w");
        assert(pfile != NULL);
        fprintf(pfile, "%f\n", A);
        for (int i = 0; i < NU; i++) {
            fprintf(pfile, "%f%c", BU(i,0), i == NU-1 ?'\n' :' ');
        }
        for (int i = 0; i < NI; i++) {
            fprintf(pfile, "%f%c", BI(i,0), i == NI-1 ?'\n' :' ');
        }
    }
}

int main() {
    srand(514);

    readMappings();
    readData();

    double A = 0;
    MatrixXd BU = MatrixXd::Zero(NU, 1);
    MatrixXd BI = MatrixXd::Zero(NI, 1);

    vector<double> tn_p(NTN, 0);
    vector<double> tt_p(NTT, 0);
    predict(tn_tuple, tn_p, NTN, A, BU, BI);
    predict(tt_tuple, tt_p, NTT, A, BU, BI);

    double last_tn_err = calERR(tn_tuple, tn_p, NTN, A, BU, BI);
    for (int time = 0; time < MAX_ITER; time++) {
        double nA = 0;
        double cA = 0;
        MatrixXd nBU = MatrixXd::Zero(NU, 1);
        MatrixXd nBI = MatrixXd::Zero(NI, 1);
        VectorXd cU = VectorXd::Zero(NU);
        VectorXd cI = VectorXd::Zero(NI);
        
        for (auto tuple : tn_tuple) {
            int user = tuple.user;
            int item = tuple.item;
            double rate = tuple.rate;
            double p = A + BU(user,0) + BI(item,0);
            double diff = (p - rate);

            nA += -diff + A;
            cA += 1;
            nBU(user,0) += -diff + BU(user,0);
            cU(user) += 1;
            nBI(item,0) += -diff + BI(item,0);
            cI(item) += 1;
        }

        if (time % 3 == 0) { 
            A = nA / cA;
        } else if (time % 3 == 1) {
            BU = nBU.array() / (cU.array() + lambda);
        } else if (time % 3 == 2) {
            BI = nBI.array() / (cI.array() + lambda);
        }

        if (time % 3 == 0) {
            predict(tn_tuple, tn_p, NTN, A, BU, BI);
            predict(tt_tuple, tt_p, NTT, A, BU, BI);

            double tn_mse = calMSE(tn_tuple, tn_p, NTN);
            double tt_mse = calMSE(tt_tuple, tt_p, NTT);
            double tn_err = calERR(tn_tuple, tn_p, NTN, A, BU, BI);
            double tt_err = calERR(tt_tuple, tt_p, NTT, A, BU, BI);
            printf("%d %f %f %f %f\n", time, tn_mse, tt_mse, tn_err, tt_err);

            if (abs(last_tn_err - tn_err) / tn_err < eps) {
                break;
            } else {
                last_tn_err = tn_err;
            }
        }
    }

    /* Save model */
    saveModel(A, BU, BI);
}
