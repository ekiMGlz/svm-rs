use nalgebra::{DMatrix, DVector, Dim, base::storage::Storage, Matrix, base::allocator::Allocator, base::default_allocator::DefaultAllocator};
use std::path::Path;
use std::collections::BTreeMap;
use quadprog_rs::base::{ConvexQP, QPOptions};
use serde::{Serialize, Deserialize};
use itertools::izip;

#[derive(Clone, Serialize, Deserialize)]
pub struct SVM {
    pub data: DMatrix<f64>,
    pub labels: DVector<f64>,
    alphas: Option<DVector<f64>>,
    coef: Option<f64>,
    pub settings: SVMSettings,
}


#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct SVMSettings {
    pub kernel: KernelFunction,
    pub tol: f64,
    pub solver_settings: QPOptions,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum KernelFunction {
    DotProduct,
    Polynomial(usize, f64),
    RBF(f64)
}

impl Default for SVMSettings {
    fn default() -> SVMSettings { 
        SVMSettings{
            kernel: KernelFunction::DotProduct,
            tol: f64::INFINITY,
            solver_settings: QPOptions::default()
        }
    }
}

impl KernelFunction {
    pub fn eval<R, C, S1, S2>(&self, x: &Matrix<f64, R, C, S1>, y: &Matrix<f64, R, C, S2>) -> f64 
    where
        R: Dim,
        C: Dim,
        S1: Storage<f64, R, C>,
        S2: Storage<f64, R, C>,
        DefaultAllocator: Allocator<f64, R, C>
    {
        match *self {
            KernelFunction::DotProduct => x.dot(y),
            KernelFunction::Polynomial(degree, c) => (x.dot(y) + c).powi(degree as i32),
            KernelFunction::RBF(gamma) => (-gamma * (x - y).norm_squared()).exp()
        }
    }
}


impl SVM {

    pub fn from_csv<P: AsRef<Path>>(m: usize, n: usize, path: P, has_headers: bool, label_col: usize, settings: SVMSettings) -> SVM {
        // Parse csv, build model
        let raw_data = f64matrix_from_csv(m, n, path, has_headers);
        let labels = raw_data.column(label_col).clone_owned();
        let data = raw_data.remove_column(label_col);


        SVM{
            data,
            labels,
            alphas: None,
            coef: None,
            settings
        }
    }

    pub fn from(data: DMatrix<f64>, labels: DVector<f64>, settings: SVMSettings) -> SVM{
        SVM{
            data,
            labels,
            alphas: None,
            coef: None,
            settings
        }
    }


    pub fn eval(&self, data: &DVector<f64>) -> f64 {
        // Evaluate new data against the model
        
        match (self.alphas.as_ref(), self.coef.as_ref()){
            (Some(alphas), Some(coef)) => {
                let mut sum = *coef;
                let data_row = data.transpose();
                for (row, alpha, y) in izip!(self.data.row_iter(), alphas.iter(), self.labels.iter()){
                    sum += alpha * y * self.settings.kernel.eval(&row, &data_row);
                }
                sum
            },
            _ => { f64::NAN }
        }
        
    }

    pub fn fit(&mut self) {
        let n = self.data.nrows();
        
        let hess = {
            let mut hess = DMatrix::<f64>::zeros(n, n);
            
            for (i, x_i) in self.data.row_iter().enumerate() {
                for  (j, x_j) in self.data.row_iter().enumerate().skip(i) {
                    hess[(i, j)] =  self.labels[i] * self.labels[j] * self.settings.kernel.eval(&x_i, &x_j);
                }
            }
            hess.fill_lower_triangle_with_upper_triangle();
            hess
        };

        let c = DVector::<f64>::repeat(n, -1.0);
        let eq_constraint: DMatrix<f64> = DMatrix::<f64>::from_iterator(1, n, self.labels.iter().map(|x|{*x}));
        
        let mut bounds: BTreeMap<usize, (Option<f64>, Option<f64>)> = BTreeMap::new();
        
        for i in 0..n{
            bounds.insert(i, (Some(0.0), Some(self.settings.tol)));
        }

        let qp = ConvexQP{
            hess,
            c,
            a_eq: Some(eq_constraint),
            b_eq: Some(DVector::<f64>::zeros(1)),
            bounds: Some(bounds),
            options: self.settings.solver_settings,
            ..Default::default()
        };

        let soln = qp.solve().unwrap();
        let alphas = soln.x;

        let mut count: usize = 0;
        
        let indices = alphas.iter().enumerate().filter(|(_, &alpha)| {
            alpha > 1e-6 && alpha < self.settings.tol * 0.98
        }).map(|(i, _)| {count += 1; i});
        
        let mut sum: f64 = 0.0;
        
        for i in indices {
            let a = self.labels[i] - self.labels[i] * alphas.dot(&qp.hess.column(i));
            sum += a;
        }

        let coef = sum/(count as f64);

        self.alphas = Some(alphas);
        self.coef = Some(coef);
    }

    pub fn get_alphas(&self) -> &Option<DVector<f64>> {
        &self.alphas
    }

    pub fn get_coef(&self) -> &Option<f64> {
        &self.coef
    }
}

fn f64matrix_from_csv<P: AsRef<Path>>(m: usize, n: usize, path: P, has_headers: bool) -> DMatrix<f64> {
    let mut reader = csv::ReaderBuilder::new()
    .has_headers(has_headers)
    .trim(csv::Trim::All)
    .from_path(path)
    .unwrap();

    let mut v = Vec::new();

    for record in reader.records().flatten() {
        v.append(&mut record.iter().map(|s| -> f64 {s.parse().unwrap()}).collect());
    }


    DMatrix::<f64>::from_vec(n, m, v).transpose()
}

mod tests {
    use super::{SVM, SVMSettings, KernelFunction};
    use nalgebra::{DMatrix, DVector};
    use std::fs::File;
    use std::io::Write;
    use rand::{distributions::Distribution, distributions::Uniform};

    #[test]
    fn dot_prod_test_set() -> Result<(), Box<dyn std::error::Error>> {
        let mut model = SVM::from_csv(50, 3, "test_sets/linear_test_set.csv", false, 2, SVMSettings::default());

        model.fit();

        let mut file = File::create("test_results/linear_model.json").unwrap();
        let j = serde_json::to_string(&model)?;
        write!(file, "{}", j)?;

        let real_boundry = |x: &DVector<f64>| -> f64 {
            x[0] + x[1] + 0.5
        };

        let mut v = DVector::<f64>::zeros(2);
        let between = Uniform::<f64>::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let mut file = File::create("test_results/linear_results.csv")?;

        let n = 100;
        for _ in 0..n {
            v[0] = between.sample(&mut rng);
            v[1] = between.sample(&mut rng);
            let real_score = real_boundry(&v);
            let model_score = model.eval(&v);
            let pass = (real_score * model_score).is_sign_positive();
            writeln!(file, "{},{},{}", real_score, model_score, pass)?;
        }

        Ok(())
    }

    #[test]
    fn poly_test_set() -> Result<(), Box<dyn std::error::Error>> {
        let settings = SVMSettings{
            kernel: KernelFunction::Polynomial(2, 1.0),
            tol: 1_000.0,
            .. Default::default()
        };

        let mut model = SVM::from_csv(50, 3, "test_sets/poly_test_set.csv", false, 2, settings);

        model.fit();

        let mut file = File::create("test_results/poly_model.json").unwrap();
        let j = serde_json::to_string(&model)?;
        write!(file, "{}", j)?;

        let real_boundry = |x: &DVector<f64>| -> f64 {
            0.2*(1.0 + 5.0*x[0] - 2.0*x[1]).powi(2) - 0.8*(1.0 - 3.0*x[0] - 4.0*x[1]).powi(2) + 9.0
        };

        let mut v = DVector::<f64>::zeros(2);
        let between = Uniform::<f64>::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let mut file = File::create("test_results/poly_results.csv")?;

        let n = 100;
        for _ in 0..n {
            v[0] = between.sample(&mut rng);
            v[1] = between.sample(&mut rng);
            let real_score = real_boundry(&v);
            let model_score = model.eval(&v);
            let pass = (real_score * model_score).is_sign_positive();
            writeln!(file, "{},{},{}", real_score, model_score, pass)?;
        }

        Ok(())
    }

    #[test]
    fn rbf_test_set() -> Result<(), Box<dyn std::error::Error>> {
        let settings = SVMSettings{
            kernel: KernelFunction::RBF(7.0),
            tol: 10_000.0,
            .. Default::default()
        };

        let mut model = SVM::from_csv(50, 3, "test_sets/rbf_test_set.csv", false, 2, settings);

        model.fit();

        let mut file = File::create("test_results/rbf_model.json").unwrap();
        let j = serde_json::to_string(&model)?;
        write!(file, "{}", j)?;

        let real_boundry = |x: &DVector<f64>| -> f64 {
            0.2*(-7.0*((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2))).exp() + 0.8*(-7.0*((x[0] + 0.2).powi(2) + (x[1] + 0.2).powi(2))).exp() - 0.1
        };

        let mut v = DVector::<f64>::zeros(2);
        let between = Uniform::<f64>::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let mut file = File::create("test_results/rbf_results.csv")?;

        let n = 100;
        for _ in 0..n {
            v[0] = between.sample(&mut rng);
            v[1] = between.sample(&mut rng);
            let real_score = real_boundry(&v);
            let model_score = model.eval(&v);
            let pass = (real_score * model_score).is_sign_positive();
            writeln!(file, "{},{},{}", real_score, model_score, pass)?;
        }

        Ok(())
    }
}