use nalgebra::{DMatrix, DVector};
use std::path::Path;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use quadprog_rs::base::{ConvexQP};
use itertools::izip;

pub struct SVM {
    vector: DVector<f64>,
    coef: f64,
    settings: SVMSettings,
}


#[derive(Copy, Clone)]
pub struct SVMSettings {
    pub kernel: KernelFunction,
    pub tol: f64,
}

#[derive(Copy, Clone)]
pub enum KernelFunction {
    DotProduct,
    Polynomial(usize, f64),
    RBF(f64)
}

impl SVM {

    pub fn from_csv<P: AsRef<Path>>(m: usize, n: usize, path: P, has_headers: bool, label_col: usize, settings: SVMSettings) -> SVM {
        // Parse csv, build model
        let raw_data = f64matrix_from_csv(m, n, path, has_headers);
        let labels = raw_data.column(label_col).clone_owned();
        let data = raw_data.remove_column(label_col);


        SVM::from(&data, &labels, settings)
    }

    pub fn from(data: &DMatrix<f64>, labels: &DVector<f64>, settings: SVMSettings) -> SVM {
        // From given data structs
        let n = data.nrows();
        
        let hess = {
            let mut hess = DMatrix::<f64>::zeros(n, n);
            
            for (i, x_i) in data.row_iter().enumerate() {
                for  (j, x_j) in data.row_iter().enumerate().skip(i) {
                    let k = match settings.kernel {
                        KernelFunction::DotProduct => x_i.dot(&x_j),
                        KernelFunction::Polynomial(degree, c) => (x_i.dot(&x_j) + c).powi(degree as i32),
                        KernelFunction::RBF(gamma) => (-gamma * (x_i - x_j).norm()).exp()
                    };

                    hess[(i, j)] = labels[i] * labels[j] * k;
                }
            }
            hess.fill_lower_triangle_with_upper_triangle();
            hess
        };

        let c = DVector::<f64>::repeat(n, 1.0);
        let eq_constraint: DMatrix<f64> = DMatrix::<f64>::from_iterator(n, 1, labels.iter().map(|x|{*x}));
        
        let mut bounds: BTreeMap<usize, (Option<f64>, Option<f64>)> = BTreeMap::new();
        
        for i in 0..n{
            bounds.insert(i, (Some(0.0), Some(settings.tol)));
        }

        let qp = ConvexQP{
            hess,
            c,
            a_eq: Some(eq_constraint),
            b_eq: Some(DVector::<f64>::zeros(1)),
            bounds: Some(bounds),
            ..Default::default()
        };

        let soln = qp.solve().unwrap();
        let alphas = soln.x;
        // Change to kernel
        let vector = (data.transpose() * &alphas).component_mul(labels);

        let mut count: usize = 0;
        // Change x_i.dot(&vec) to K(x_i, vector)
        let sum: f64 = izip!(alphas.iter(), data.row_iter(), labels.iter())
                            .filter(|(&alpha, _, _)| { alpha.is_sign_positive() && alpha.is_normal() && 
                                (alpha - settings.tol).is_sign_positive() && (alpha - settings.tol).is_normal()})
                            .map(|(alpha, x_i, y_i)| {count += 1; - alpha * y_i * x_i.dot(&vector)}).sum();
        
        let coef = sum / count as f64;
        
        SVM {
            vector,
            coef,
            settings: settings,
        }
    }

    fn eval(&self, data: DVector<f64>) -> f64 {
        // Evaluate a new data against the model
        0.0
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