#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]

extern crate nalgebra as na;
use na::*;
use std::f64::consts::PI;

type Vector = DVector<f64>;
type Matrix = DMatrix<f64>;

// Heat equation
// u_t = gamma * u_xx, 0<x<1, t>0
// u(t,0) = alpha, u(t,1) = beta
// u(0,x) = f(x)

struct HeatEquation<F> where F: Fn(f64) -> f64 {
  gamma: f64,
  alpha: f64,
  beta: f64,
  f: F
}

#[derive(Copy, Clone)]
enum HeatMethod {
  Explicit,
  Implicit,
  CrankNicolson
}

impl<F> HeatEquation<F> where F: Fn(f64) -> f64 {
  fn finite_differences(&self, method: HeatMethod, dx: f64, dt: f64, t: f64) -> Vector {
    let n = (1.0/dx) as usize + 1;
    let mu = self.gamma*dt/(dx*dx);

    let mut b = Vector::from_element(n, 0.0);
    b[0] = mu * self.alpha;
    b[n-1] = mu * self.beta;


    let next_iterate = |u| match method {
      HeatMethod::Explicit => {
        let H = &tridiagonal(n, 1.0 - 2.0 * mu, mu);

        H * u + &b
      ,
      HeatMethod::Implicit => {
        let H_hat = tridiagonal(n, 1.0 + 2.0 * mu, -mu);
        let H_hat_inverse = &H_hat.try_inverse().unwrap();
    
        H_hat_inverse * (u + &b)
      },
      HeatMethod::CrankNicolson => {
        let B_hat = tridiagonal(n, 1.0 + mu, -0.5*mu);
        let B_hat_inverse = &B_hat.try_inverse().unwrap();
        let B = &tridiagonal(n, 1.0 - mu, 0.5*mu);
    
        B_hat_inverse * (B*u + &b)
      }
    };}

    let iterations = (t/dt) as i64;

    let mut u = vectorize(&self.f,n).clone();
    for _ in 0..iterations {
      u = next_iterate(u);
    }

    u
  }
}









// u_tt = c^2 u_xx + force(t,x), 0<x<1, t>0
// u(t,0) = alpha, u(t,1) = beta
// u(0,x) = f(x), u_t(0,x) = g(x)
struct WaveEquation<F,G,H> where F: Fn(f64) -> f64, G: Fn(f64) -> f64, H: Fn(f64, f64) -> f64 {
  c: f64,
  alpha: f64, beta: f64, 
  f: F, g: G,
  force: H
}

fn tridiagonal(n: usize, diagonal: f64, off: f64) -> Matrix {
  let mut A = Matrix::from_diagonal_element(n, n, diagonal);
  for i in 0..n-1 {
    A[(i+1,i)] = off;
    A[(i,i+1)] = off;
  }
  A
}

fn vectorize<F>(f: &F, n: usize) -> Vector where F: Fn(f64) -> f64 {
  let mut v = Vector::from_element(n, 0.0);
  for i in 0..n {
    let x = (i as f64)/((n-1) as f64);
    v[i] = f(x);
  }
  v
}

impl<F,G,H> WaveEquation<F,G,H> where F: Fn(f64) -> f64, G: Fn(f64) -> f64, H: Fn(f64, f64) -> f64 {
  fn finite_differences(&self, dx: f64, dt: f64, t: f64) -> Vector {
    let n = (1.0/dx) as usize + 1;
    let f = vectorize(&self.f, n);
    let g = vectorize(&self.g, n);

    let sigma = self.c*dt/dx;
    let B = &tridiagonal(n, 2.0*(1.0-sigma*sigma), sigma*sigma);

    let mut b = Vector::from_element(n, 0.0);
    b[0] = sigma * sigma * self.alpha;
    b[n-1] = sigma * sigma * self.beta;

    let mut u_previous = f;
    let mut u_current = 0.5*B*&u_previous + g*dt + 0.5*&b;
    // Minus one because u_current is already one time step ahead of 0
    let iterations = (t/dt) as i64 - 1;
    let mut t_current = dt;

    for _ in 0..iterations {
      let force_vec = vectorize(&|x| (self.force)(t_current,x), n);
      let u_next = B * &u_current - &u_previous + &b + force_vec * dt * dt;

      u_previous = u_current;
      u_current  = u_next;
      t_current += dt;
    }

    u_current
  }
}

fn main() {
  let equation = WaveEquation {
    c: 0.25,
    alpha: 0.0, beta: 0.0,
    f: |x| 0.0,
    g: |x| 0.0,
    force: |t,x| 3.0*(x-0.5).signum()*(PI*t).sin()
  };

  let (dx,dt) = (0.01, 0.01);
  let times = [0.0, 0.25, 0.4, 0.5, 0.75, 1.0];
  let data = times.map(|t| (t, equation.finite_differences(dx, dt, t))).to_vec();
  plot("Wave", &data);

  println!("{}", equation.finite_differences(dx,dt,1.0))
}

fn plot(title: &str, data: &Vec<(f64, Vector)>) {
  use plotters::prelude::*;

  let width = 600;
  let height = 400;
  let filename = format!("plots/{title}.png");
  let area = BitMapBackend::new(filename.as_str(), (width, height))
    .into_drawing_area();

  area.fill(&WHITE).unwrap();

  // note: iamax returns the index of the element of largest absolute value
  let absmax = |u: &Vector| u[u.iamax()].abs();

  let M = data.iter().map(|(t,u)| absmax(u))
    .fold(f64::NEG_INFINITY, f64::max);

  let mut chart = ChartBuilder::on(&area)
    .set_label_area_size(LabelAreaPosition::Left, 40)
    .set_label_area_size(LabelAreaPosition::Bottom, 40)
    .caption(title, ("sans-serif", 40))
    .build_cartesian_2d(0.0..1.0, -M..M)
    .unwrap();

  chart.configure_mesh().draw().unwrap();

  let colors = [BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW];

  for (index, (t,u)) in data.iter().enumerate() {
    let N: usize = u.len();
    let points = (0..N).map(|i| ((i as f64)/((N-1) as f64), u[i]));
    let color = colors[index % colors.len()];

    chart.draw_series(LineSeries::new(points, color))
      .unwrap()
      .label(format!("t={t}"))
      .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
  }

  chart.configure_series_labels()
    .border_style(&BLACK)
    .background_style(&WHITE.mix(0.8))
    .draw()
    .unwrap();

}

