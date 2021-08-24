using System;
using System.Collections.Generic;
using System.Linq;
using Skender.Stock.Indicators;

namespace RFFARIMA
{
    class Pandas
    {
        public static Dictionary<string, List<string>> read_csv(string path, bool header = true)
        {
            Dictionary<string, List<string>> df = new Dictionary<string, List<string>>();
            if (header == true)
            {
                string[] lines = System.IO.File.ReadAllLines(@path);
                string[] headers = lines[0].Split(';');
                foreach (string sub in headers)
                {
                    df.Add(sub, new List<string>());
                }
                for (int i = 1; i < lines.Length; i++)
                {
                    string[] substring = lines[i].Split(';');
                    for (int j = 0; j < substring.Length; j++)
                    {
                        df[headers[j]].Add(substring[j]);
                    }
                }
                return df;
            }
            else
            {
                string[] lines = System.IO.File.ReadAllLines(@path);
                for (int i = 1; i < lines[0].Split(";").Length; i++)
                {
                    df.Add(i.ToString(), new List<string>());
                }
                for (int i = 0; i < lines.Length; i++)
                {
                    string[] substring = lines[i].Split(';');
                    for (int j = 1; j < substring.Length; j++)
                    {
                        df[j.ToString()].Add(substring[j]);
                    }
                }
                return df;
            }
        }
    }
    class LinearAlgebra
    {
        public static double[][] MatrixCreate(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
        public static double[][] MatrixProduct(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;
            if (aCols != bRows)
                throw new Exception("Non-conformable matrices in MatrixProduct");
            double[][] result = MatrixCreate(aRows, bCols);
            for (int i = 0; i < aRows; ++i) // каждая строка A
                for (int j = 0; j < bCols; ++j) // каждый столбец B
                    for (int k = 0; k < aCols; ++k)
                        result[i][j] += matrixA[i][k] * matrixB[k][j];
            return result;
        }
        public static double[][] MatrixDuplicate(double[][] matrix)
        {
            double[][] result = MatrixCreate(matrix.Length, matrix[0].Length);
            for (int i = 0; i < matrix.Length; ++i)
                for (int j = 0; j < matrix[i].Length; ++j)
                    result[i][j] = matrix[i][j];
            return result;
        }
        public static double[][] MatrixTranspose(double[][] matrix)
        {
            double[][] result = MatrixCreate(matrix[0].Length, matrix.Length);
            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[i].Length; j++)
                    result[j][i] = matrix[i][j];
            return result;
        }
        public static double[][] MatrixDecompose(double[][] matrix, out int[] perm, out int toggle)
        {
            int n = matrix.Length; // для удобства
            double[][] result = MatrixDuplicate(matrix);
            perm = new int[n];
            for (int i = 0; i < n; ++i) { perm[i] = i; }
            toggle = 1;
            for (int j = 0; j < n - 1; ++j) // каждый столбец
            {
                double colMax = Math.Abs(result[j][j]); // Наибольшее значение в столбце j
                int pRow = j;
                for (int i = j + 1; i < n; ++i)
                {
                    if (result[i][j] > colMax)
                    {
                        colMax = result[i][j];
                        pRow = i;
                    }
                }
                if (pRow != j) // перестановка строк
                {
                    double[] rowPtr = result[pRow];
                    result[pRow] = result[j];
                    result[j] = rowPtr;
                    int tmp = perm[pRow]; // Меняем информацию о перестановке
                    perm[pRow] = perm[j];
                    perm[j] = tmp;
                    toggle = -toggle; // переключатель перестановки строк
                }
                if (Math.Abs(result[j][j]) < 1.0E-20)
                    return null;
                for (int i = j + 1; i < n; ++i)
                {
                    result[i][j] /= result[j][j];
                    for (int k = j + 1; k < n; ++k)
                        result[i][k] -= result[i][j] * result[j][k];
                }
            } // основной цикл по столбцу j
            return result;
        }
        public static double[] HelperSolve(double[][] luMatrix, double[] b)
        {
            int n = luMatrix.Length;
            double[] x = new double[n];
            b.CopyTo(x, 0);
            for (int i = 1; i < n; ++i)
            {
                double sum = x[i];
                for (int j = 0; j < i; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum;
            }
            x[n - 1] /= luMatrix[n - 1][n - 1];
            for (int i = n - 2; i >= 0; --i)
            {
                double sum = x[i];
                for (int j = i + 1; j < n; ++j)
                    sum -= luMatrix[i][j] * x[j];
                x[i] = sum / luMatrix[i][i];
            }
            return x;
        }
        public static double[][] MatrixInverse(double[][] matrix)
        {
            int n = matrix.Length;
            double[][] result = MatrixDuplicate(matrix);
            int[] perm;
            int toggle;
            double[][] lum = MatrixDecompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new Exception("Unable to compute inverse");
            double[] b = new double[n];
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == perm[j])
                        b[j] = 1.0;
                    else
                        b[j] = 0.0;
                }
                double[] x = HelperSolve(lum, b);
                for (int j = 0; j < n; ++j)
                    result[j][i] = x[j];
            }
            return result;
        }
    }
    class Econometrics
    {
        public static double NormalDist(double mean, double std)
        {
            Random rand = new Random();
            double u1 = 1.0 - rand.NextDouble(); // uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
            return mean + std * randStdNormal; // random normal(mean,stdDev^2)
        }
        public static double UniformPiDist()
        {
            Random rand = new Random();
            double u1 = 0.5 - rand.NextDouble(); // uniform(0,1] random doubles
            return 2 * u1 * Math.PI;
        }
        public static double Median(List<double> numbers)
        {
            int numberCount = numbers.Count();
            int halfIndex = numbers.Count() / 2;
            var sortedNumbers = numbers.OrderBy(n => n);
            double median;
            if ((numberCount % 2) == 0)
            {
                median = ((sortedNumbers.ElementAt(halfIndex) +
                    sortedNumbers.ElementAt((halfIndex - 1))) / 2);
            }
            else
            {
                median = sortedNumbers.ElementAt(halfIndex);
            }
            return median;
        }
        public static double Model(List<double> initial, List<double> br, List<double> si, int AR, int features = 200)
        {
            List<double> result = new List<double>(); // price movement
            List<double> result_br = new List<double>();
            List<double> result_si = new List<double>();
            for (int i = 1; i < initial.Count; i++)
            {
                result.Add((initial[i] - initial[i - 1]) / initial[i - 1]); // adding price deltas
                result_br.Add((br[i] - br[i - 1]) / br[i - 1]);
                result_si.Add((si[i] - si[i - 1]) / si[i - 1]);
            }
            double[][] X = LinearAlgebra.MatrixCreate(result.Count - AR, 3 * AR + 2); // feature matrix
            double[][] Xp = LinearAlgebra.MatrixCreate(1, 3 * AR + 2);
            double[][] y = LinearAlgebra.MatrixCreate(result.Count - AR, 1);
            int RSI = 30;
            double sum_up = 0;
            double sum_down = 0;
            // no questions up to this point
            for (int i = RSI; i < result.Count; i++) // iter through the time series
            {
                X[i - RSI][0] = 1; // constant term
                for (int j = 1; j <= AR; j++)
                {
                    X[i - RSI][j] = result[i - j]; // AR
                    X[i - RSI][j + AR] = result_br[i - j];
                    X[i - RSI][j + 2 * AR] = result_si[i - j];
                }
                sum_up = 0;
                sum_down = 0;
                for (int j = 1; j <= RSI; j++)
                {
                    if (result[i - j] > 0)
                    {
                        sum_up = sum_up + result[i - j];
                    }
                    else
                    {
                        sum_down = sum_down + result[i - j];
                    }
                }
                sum_up = sum_up / RSI;
                sum_down = (-1) * sum_down / RSI;
                X[i - RSI][3 * AR + 1] = sum_up / sum_down;
                y[i - RSI][0] = result[i];
            }

            Xp[0][0] = 1;
            for (int j = 1; j <= AR; j++)
            {
                Xp[0][j] = result[result.Count - j];
                Xp[0][j + AR] = result_br[result_br.Count - j];
                Xp[0][j + 2 * AR] = result_si[result_si.Count - j];
            }
            sum_up = 0;
            sum_down = 0;
            for (int j = 1; j <= RSI; j++)
            {
                if (result[result.Count - j] > 0)
                {
                    sum_up = sum_up + result[result.Count - j];
                }
                else
                {
                    sum_down = sum_down + result[result.Count - j];
                }
            }
            sum_up = sum_up / RSI;
            sum_down = (-1) * sum_down / RSI;
            Xp[0][3 * AR + 1] = sum_up / sum_down;
            // finished with filling X and y
            List<double> res_list = new List<double>();
            for (int i = 0; i < X.GetLength(0); i++)
            {
                for (int j = 0; j < X.GetLength(0); j++)
                {
                    double res_sum = 0;
                    for (int k = 0; k < X[0].GetLength(0); k++)
                    {
                        res_sum += Math.Pow(X[i][k] - X[j][k], 2);
                    }
                    res_list.Add(res_sum);
                }
            }
            double sigma2 = Median(res_list);
            double[][] w = LinearAlgebra.MatrixCreate(X[0].GetLength(0), features);
            double[][] b = LinearAlgebra.MatrixCreate(1, features);
            double[][] ones = LinearAlgebra.MatrixCreate(X.GetLength(0), 1);
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w[0].GetLength(0); j++)
                {
                    w[i][j] = NormalDist(0, Math.Sqrt(1 / sigma2));
                }
            }
            for (int i = 0; i < features; i++)
            {
                b[0][i] = UniformPiDist();
            }

            double[][] X_mod = LinearAlgebra.MatrixCreate(X.GetLength(0), features);
            double[][] Xp_mod = LinearAlgebra.MatrixCreate(1, features);
            double[][] v1 = LinearAlgebra.MatrixProduct(X, w);
            double[][] v2 = LinearAlgebra.MatrixProduct(ones, b);
            double[][] vp1 = LinearAlgebra.MatrixProduct(Xp, w);
            for (int i = 0; i < X_mod.GetLength(0); i++)
            {
                for (int j = 0; j < X_mod[0].GetLength(0); j++)
                {
                    X_mod[i][j] = Math.Cos(v1[i][j] + v2[i][j]);
                }
            }
            for (int i = 0; i < features; i++)
            {
                Xp_mod[0][i] = Math.Cos(vp1[0][i] + b[0][i]);
            }
            double[][] weights = LinearAlgebra.MatrixCreate(features, 1); // list of weights
            try
            {
                weights = LinearAlgebra.MatrixProduct(LinearAlgebra.MatrixProduct(LinearAlgebra.MatrixInverse(LinearAlgebra.MatrixProduct(LinearAlgebra.MatrixTranspose(X_mod), X_mod)), LinearAlgebra.MatrixTranspose(X_mod)), y);
            }
            catch
            {
                for (int i = 0; i < weights.Count(); i++)
                {
                    weights[i][0] = 0;
                }
            }
            double pred = 0;
            for (int i = 0; i < weights.Count(); i++)
            {
                pred = pred + weights[i][0] * Xp_mod[0][i];
            }
            return pred;
        }
    }
    class Program
    {
        static Tuple<List<double>, List<double>> data_converter(string file, int interval, int range)
        {
            Dictionary<string, List<string>> df = Pandas.read_csv(@"C:\Users\User\Desktop\" + file);
            List<double> price_data = df["<OPEN>"].ConvertAll(double.Parse);
            List<double> final = new List<double>();
            for (int i = 0; i < price_data.Count(); i++)
            {
                if (i % interval == 0)
                {
                    final.Add(price_data[i]);
                }
            }
            List<double> train = final.GetRange(0, range);
            List<double> test = final.GetRange(range, final.Count - range);
            Tuple<List<double>, List<double>> ret = new Tuple<List<double>, List<double>>(train, test);
            return ret;
        }
        static void Main(string[] args)
        {
            int AR = 8;
            int counter = 0;
            int total = 0;

            Tuple<List<double>, List<double>> data = data_converter("Si.csv", 2, 500);
            List<double> train = data.Item1;
            List<double> test = data.Item2;

            data = data_converter("RTS.csv", 2, 500);
            List<double> rts_train = data.Item1;
            List<double> rts_test = data.Item2;

            data = data_converter("BR.csv", 2, 500);
            List<double> br_train = data.Item1;
            List<double> br_test = data.Item2;

            for (int i = 0; i < test.Count(); i++)
            {
                double pred = Econometrics.Model(train, br_train, rts_train, AR);

                if (Math.Abs(pred) > 0)
                {
                    if (Math.Sign(test[i] - train[train.Count - 1]) == Math.Sign(pred))
                    {
                        counter = counter + 1;
                    }
                    total = total + 1;
                }
                train.RemoveAt(0);
                train.Add(test[i]);
                rts_train.RemoveAt(0);
                rts_train.Add(rts_test[i]);
                br_train.RemoveAt(0);
                br_train.Add(br_test[i]);
                Console.WriteLine((double)counter / (double)total);
            }
        }
    }
}
