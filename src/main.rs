use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

const FFT_SIZE: usize = 16384;
const NUM_BINS: usize = 1600;
const SPEC_ROWS: usize = 800;
const FREQ_MIN: f32 = 40.0;
const FREQ_MAX: f32 = 4000.0;
const MAX_HARMONIC: u32 = 16;

// ---------------------------------------------------------------------------
// FFT setup
// ---------------------------------------------------------------------------

struct BinMap { idx: usize, frac: f32 }

struct FftSetup {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    bin_map: Vec<BinMap>,
    bin_res: f32,
    in_buf: Mutex<Vec<f32>>,
    out_buf: Mutex<Vec<Complex<f32>>>,
}

impl FftSetup {
    fn new(sample_rate: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let fft_clone = Arc::clone(&fft);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos())).collect();

        let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (NUM_BINS - 1) as f32);
        let bin_res = sample_rate / FFT_SIZE as f32;
        let bin_map: Vec<_> = (0..NUM_BINS).map(|i| {
            let f = FREQ_MIN * ratio.powi(i as i32);
            let exact = f / bin_res;
            let idx = (exact as usize).min(FFT_SIZE / 2 - 1);
            BinMap { idx, frac: exact - idx as f32 }
        }).collect();

        Self { fft: fft_clone, window, bin_map, bin_res,
               in_buf: Mutex::new(fft.make_input_vec()),
               out_buf: Mutex::new(fft.make_output_vec()) }
    }

    fn process_frame(&self, samples: &[f32], out: &mut [f32]) {
        let mut ib = self.in_buf.lock();
        let mut ob = self.out_buf.lock();
        for (d, (&s, &w)) in ib.iter_mut().zip(samples.iter().zip(self.window.iter())) { *d = s * w; }
        let _ = self.fft.process(&mut *ib, &mut *ob);
        for (o, bm) in out.iter_mut().zip(self.bin_map.iter()) {
            let c0 = ob[bm.idx]; let m0 = (c0.re * c0.re + c0.im * c0.im).sqrt();
            *o = if bm.frac > 0.0 && bm.idx + 1 < ob.len() {
                let c1 = ob[bm.idx + 1]; let m1 = (c1.re * c1.re + c1.im * c1.im).sqrt();
                m0 * (1.0 - bm.frac) + m1 * bm.frac
            } else { m0 };
        }
    }

    fn mag_at_freq(&self, freq: f32, mags: &[f32]) -> f32 {
        let exact = freq / self.bin_res;
        let idx = (exact as usize).min(FFT_SIZE / 2 - 1);
        if idx >= mags.len().saturating_sub(1) { return *mags.last().unwrap_or(&0.0); }
        let frac = exact - idx as f32;
        if frac > 0.0 { mags[idx] * (1.0 - frac) + mags[idx + 1] * frac } else { mags[idx] }
    }

    fn harmonic_mags(&self, f0: f32, mags: &[f32]) -> Vec<f32> {
        (2..=MAX_HARMONIC).map(|n| self.mag_at_freq(f0 * n as f32, mags)).collect()
    }
}

// ---------------------------------------------------------------------------
// colour ramp
// ---------------------------------------------------------------------------

fn spec_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.003 { return egui::Color32::BLACK; }
    let s: [(f32, (u8,u8,u8)); 7] = [
        (0.00,(0,0,0)),(0.08,(2,2,35)),(0.20,(22,4,80)),(0.35,(100,6,28)),
        (0.50,(175,45,12)),(0.70,(230,175,10)),(1.00,(255,250,235))];
    for i in 0..s.len()-1 { let (t0,c0)=s[i]; let (t1,c1)=s[i+1];
        if t>=t0&&t<=t1 { let u=((t-t0)/(t1-t0)).clamp(0.,1.);
            return egui::Color32::from_rgb(
                (c0.0 as f32 + (c1.0 as i16 - c0.0 as i16) as f32 * u) as u8,
                (c0.1 as f32 + (c1.1 as i16 - c0.1 as i16) as f32 * u) as u8,
                (c0.2 as f32 + (c1.2 as i16 - c0.2 as i16) as f32 * u) as u8); }}
    egui::Color32::WHITE
}

fn draw_cbar(p: &egui::Painter, r: egui::Rect) {
    let h = r.height(); if h<2.{return;}
    for i in 0..20 { let t=i as f32/19.;
        let y0=r.bottom()-i as f32/20.*h; let y1=r.bottom()-(i+1)as f32/20.*h;
        p.rect_filled(egui::Rect::from_min_max(egui::pos2(r.left(),y1),egui::pos2(r.right(),y0)),0.,spec_color(t));
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn estimate_f0(mags: &[f32], freqs: &[f32]) -> Option<f32> {
    let s = freqs.iter().position(|&f| f>=60.)?;
    let e = freqs.iter().position(|&f| f>300.).unwrap_or(freqs.len());
    if e<=s { return None; }
    let mi = (s..e).max_by(|&a,&b| mags[a].partial_cmp(&mags[b]).unwrap_or(std::cmp::Ordering::Equal))?;
    (mags[mi] > 1e-6).then(|| freqs[mi])
}

fn freq_to_y(f: f32, r: &egui::Rect) -> f32 {
    let n = (f.ln()-FREQ_MIN.ln())/(FREQ_MAX.ln()-FREQ_MIN.ln());
    r.top() + (1.-n.clamp(0.,1.)) * r.height()
}

fn find_peaks(m: &[f32]) -> Vec<(usize,f32)> {
    let mut v = Vec::new();
    for i in 1..m.len().saturating_sub(1) { let x=m[i]; if x>m[i-1]&&x>=m[i+1]&&x>1e-6 { v.push((i,x)); }}
    v
}

fn label_harmonics(p: &[(usize,f32)], freqs: &[f32], f0: f32) -> Vec<(usize,u32,f32)> {
    p.iter().filter_map(|&(b,m)| {
        let f = freqs[b]; let n = (f/f0).round() as u32;
        if n<2||n>MAX_HARMONIC { return None; }
        let hf = n as f32 * f0;
        ((f-hf).abs()/hf < 0.06).then(|| (b,n,m))
    }).collect()
}

// ---------------------------------------------------------------------------
// drone
// ---------------------------------------------------------------------------

struct DroneState { enabled: bool, frequency: f32, amplitude: f32, phase: f32 }
impl DroneState { fn new() -> Self { Self { enabled: false, frequency: 110., amplitude: 0.3, phase: 0. }}}

// ---------------------------------------------------------------------------
// waterfall
// ---------------------------------------------------------------------------

struct Waterfall {
    buf: Vec<f32>, pos: usize, filled: bool, running_max: f32, tex: Option<egui::TextureHandle>,
}

impl Waterfall {
    fn new() -> Self {
        Self { buf: vec![0.; SPEC_ROWS*NUM_BINS], pos: 0, filled: false, running_max: 1e-6, tex: None }
    }
    fn push(&mut self, m: &[f32]) {
        let b=self.pos*NUM_BINS; self.buf[b..b+NUM_BINS].copy_from_slice(m);
        self.pos=(self.pos+1)%SPEC_ROWS; if self.pos==0 { self.filled=true; }
    }
    fn image(&mut self, peek: &[f32]) -> egui::ColorImage {
        let mut pk = self.running_max;
        for &v in &self.buf { if v>pk { pk=v; }} for &v in peek { if v>pk { pk=v; }}
        self.running_max = pk*0.999 + self.running_max*0.001;
        if self.running_max<1e-6 { self.running_max=1e-6; }
        let mx = if self.running_max>0.{20.*self.running_max.log10()}else{-60.};
        let fl = mx-60.;
        let mut px = Vec::with_capacity(SPEC_ROWS*NUM_BINS);
        if self.filled {
            for i in 0..SPEC_ROWS { let row=(self.pos+i)%SPEC_ROWS; let b=row*NUM_BINS;
                for bin in 0..NUM_BINS { let m=self.buf[b+bin];
                    let db = if m>0.{20.*m.log10()}else{-100.};
                    let n=((db-fl)/(mx-fl)).clamp(0.,1.); px.push(spec_color(n)); }}
        } else {
            for _ in self.pos..SPEC_ROWS { for _ in 0..NUM_BINS { px.push(egui::Color32::BLACK); }}
            for i in 0..self.pos { let b=i*NUM_BINS;
                for bin in 0..NUM_BINS { let m=self.buf[b+bin];
                    let db = if m>0.{20.*m.log10()}else{-100.};
                    let n=((db-fl)/(mx-fl)).clamp(0.,1.); px.push(spec_color(n)); }}
        }
        egui::ColorImage::new([NUM_BINS,SPEC_ROWS],px)
    }
}

// ---------------------------------------------------------------------------
// app
// ---------------------------------------------------------------------------

struct VoiceHarmApp {
    audio_buf: Arc<Mutex<Vec<f32>>>,
    fft_setup: FftSetup,
    waterfall: Waterfall,
    freqs: Vec<f32>,
    current_mags: Vec<f32>,
    drone_state: Arc<Mutex<DroneState>>,
    drone_on: bool,
    drone_vol: f32,
}

impl VoiceHarmApp {
    fn new(sr: f32, audio: Arc<Mutex<Vec<f32>>>, drone: Arc<Mutex<DroneState>>) -> Self {
        let r = (FREQ_MAX/FREQ_MIN).powf(1./(NUM_BINS-1)as f32);
        let fs: Vec<_> = (0..NUM_BINS).map(|i| FREQ_MIN*r.powi(i as i32)).collect();
        Self { audio_buf: audio, fft_setup: FftSetup::new(sr), waterfall: Waterfall::new(),
               freqs: fs, current_mags: vec![0.;NUM_BINS], drone_state: drone, drone_on: false, drone_vol: 0.3 }
    }
}

impl eframe::App for VoiceHarmApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.ctx().set_visuals(egui::Visuals::dark());
        // ── audio processing (same as before) ──
        let sample = { let b = self.audio_buf.lock(); (b.len()>=FFT_SIZE).then(|| b[b.len()-FFT_SIZE..].to_vec()) };
        if let Some(s) = sample {
            { let mut b = self.audio_buf.lock(); let l = b.len();
              if l > FFT_SIZE*3 { b.drain(0..l - FFT_SIZE*2); }}
            self.fft_setup.process_frame(&s, &mut self.current_mags);
            self.waterfall.push(&self.current_mags);
        }

        let f0 = estimate_f0(&self.current_mags, &self.freqs);
        if let Some(f) = f0 { self.drone_state.lock().frequency = f; }

        // ── top panel: controls ──
        egui::containers::Panel::top("top")
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(15, 15, 25)))
            .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);

                // F0
                if let Some(f) = f0 {
                    ui.colored_label(egui::Color32::WHITE, format!("F0: {:.1} Hz", f));
                } else {
                    ui.colored_label(egui::Color32::DARK_GRAY, "F0: --");
                }

                ui.separator();

                // Drone toggle button
                let drone_label = egui::RichText::new("Drone").color(egui::Color32::WHITE);
                let btn = if self.drone_on {
                    egui::Button::new(drone_label).fill(egui::Color32::from_rgb(25, 160, 25))
                } else {
                    egui::Button::new(drone_label)
                };
                if ui.add(btn).clicked() {
                    self.drone_on = !self.drone_on;
                    self.drone_state.lock().enabled = self.drone_on;
                }

                // Volume slider
                ui.add(egui::Slider::new(&mut self.drone_vol, 0.0..=1.0)
                    .show_value(false)
                    .text("Vol"));
                ui.label(format!("{:.0}%", self.drone_vol * 100.0));
                self.drone_state.lock().amplitude = self.drone_vol;
            });
        });

        // ── right panel: harmonic profile ──
        egui::containers::Panel::right("profile")
            .resizable(false)
            .default_size(72.0)
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(15, 15, 25)))
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new("H").size(11.0).color(egui::Color32::DARK_GRAY));
                });

                if let Some(f) = f0 {
                    let hm = self.fft_setup.harmonic_mags(f, &self.current_mags);
                    let mx = hm.iter().cloned().max_by(|a,b|a.partial_cmp(b).unwrap()).unwrap_or(1e-6).max(1e-6);
                    for (i, &m) in hm.iter().enumerate() {
                        let n = i + 2;
                        ui.horizontal(|ui| {
                            ui.set_height(12.0);
                            ui.label(egui::RichText::new(format!("{}", n)).size(10.0).color(egui::Color32::from_gray(140)));
                            let frac = (m / mx).clamp(0.0, 1.0);
                            let pb = egui::ProgressBar::new(frac).desired_width(ui.available_width().max(8.0));
                            ui.add(pb);
                        });
                    }
                } else {
                    ui.label(egui::RichText::new("no F0").size(10.0).color(egui::Color32::from_gray(50)));
                }
            });

        // ── central panel: spectrogram canvas ──
        egui::CentralPanel::default().show(ui, |ui| {
            let painter = ui.painter();
            let canvas = ui.max_rect();
            painter.rect_filled(canvas, 0.0, egui::Color32::BLACK);

            // split canvas: freq labels | spectrogram | colour bar
            let label_w = 44.0;
            let cbar_w = 16.0;
            let gap = 4.0;
            let spec_rect = egui::Rect::from_min_size(
                egui::pos2(canvas.left() + label_w, canvas.top()),
                egui::vec2((canvas.width() - label_w - cbar_w - gap).max(64.0), canvas.height()),
            );
            let cbar_rect = egui::Rect::from_min_size(
                egui::pos2(spec_rect.right() + gap, canvas.top()),
                egui::vec2(cbar_w, canvas.height()),
            );

            // ── texture ──
            let img = self.waterfall.image(&self.current_mags);
            let tex = self.waterfall.tex.get_or_insert_with(||
                ui.ctx().load_texture("wf", img.clone(), egui::TextureOptions::NEAREST));
            tex.set(img, egui::TextureOptions::NEAREST);
            painter.image(tex.id(), spec_rect,
                egui::Rect::from_min_max(egui::pos2(0.,0.), egui::pos2(1.,1.)), egui::Color32::WHITE);

            // ── freq labels + grid ──
            let lf: [f32;8] = [50.,100.,200.,500.,1000.,2000.,3000.,4000.];
            for &f in &lf { if f<FREQ_MIN||f>FREQ_MAX { continue; }
                let x = spec_rect.left() + (f.ln()-FREQ_MIN.ln())/(FREQ_MAX.ln()-FREQ_MIN.ln()) * spec_rect.width();
                let lb = if f>=1000. { format!("{}k",(f/1000.)as u32) } else { format!("{f}") };
                painter.text(egui::pos2(x,canvas.bottom()+8.),egui::Align2::CENTER_TOP,&lb,
                    egui::FontId::proportional(11.),egui::Color32::GRAY);
                painter.line_segment([egui::pos2(x,canvas.top()),egui::pos2(x,canvas.bottom())],
                    egui::Stroke::new(1.,egui::Color32::from_rgba_premultiplied(80,80,80,18)));
            }

            // ── colour bar ──
            draw_cbar(&painter, cbar_rect);
            painter.text(egui::pos2(cbar_rect.right()+2.,cbar_rect.top()),egui::Align2::LEFT_TOP,
                "0",egui::FontId::proportional(9.),egui::Color32::GRAY);
            painter.text(egui::pos2(cbar_rect.right()+2.,cbar_rect.bottom()),egui::Align2::LEFT_BOTTOM,
                "-60",egui::FontId::proportional(9.),egui::Color32::GRAY);

            // ── "now" ──
            painter.text(egui::pos2(spec_rect.left(),canvas.bottom()+8.),egui::Align2::LEFT_TOP,
                "now",egui::FontId::proportional(10.),egui::Color32::from_gray(100));

            // ── harmonic number labels ──
            if let Some(f) = f0 {
                let peaks = find_peaks(&self.current_mags);
                for &(_b,n,_m) in &label_harmonics(&peaks,&self.freqs,f) {
                    let ff = (n as f32*f).max(FREQ_MIN).min(FREQ_MAX);
                    let y = freq_to_y(ff, &spec_rect);
                    let tag = format!("{}",n);
                    let tr = egui::Rect::from_min_size(egui::pos2(spec_rect.left()-16.,y-7.),egui::vec2(16.,14.));
                    painter.rect_filled(tr,2.,egui::Color32::from_rgb(60,60,80));
                    painter.text(tr.center(),egui::Align2::CENTER_CENTER,&tag,
                        egui::FontId::proportional(10.),egui::Color32::from_rgb(220,220,200));
                    painter.line_segment([egui::pos2(spec_rect.left(),y),egui::pos2(spec_rect.right(),y)],
                        egui::Stroke::new(1.,egui::Color32::from_rgba_premultiplied(255,255,200,25)));
                }
            }

            // ── cursor overlay ──
            if let Some(mp) = ui.ctx().pointer_hover_pos() {
                if spec_rect.contains(mp) {
                    let xn = ((mp.x-spec_rect.left())/spec_rect.width()).clamp(0.,1.);
                    let yn = ((mp.y-canvas.top())/canvas.height()).clamp(0.,1.);
                    let freq = FREQ_MIN*(FREQ_MAX/FREQ_MIN).powf(xn);
                    let bin = (xn*(NUM_BINS-1)as f32).round() as usize;
                    let mag = self.current_mags.get(bin).copied().unwrap_or(0.);
                    let db = if mag>0.{20.*mag.log10()}else{-100.};
                    let t_sec = (1.-yn)*(SPEC_ROWS as f32/60.);

                    let cc = egui::Color32::from_rgba_premultiplied(200,200,200,70);
                    painter.line_segment([egui::pos2(mp.x,canvas.top()),egui::pos2(mp.x,canvas.bottom())],
                        egui::Stroke::new(1.,cc));
                    painter.line_segment([egui::pos2(spec_rect.left(),mp.y),egui::pos2(spec_rect.right(),mp.y)],
                        egui::Stroke::new(1.,cc));

                    let fs = if freq>=1000.{format!("{:.1}kHz",freq/1000.)}else{format!("{:.1}Hz",freq)};
                    let info = format!("{fs}\n{db:.1}dB\n-{t_sec:.1}s");
                    let font = egui::FontId::monospace(13.);
                    let g = painter.layout_no_wrap(info,font,egui::Color32::WHITE);
                    let pad = 5.; let sz = egui::vec2(g.size().x+pad*2.,g.size().y+pad*2.);
                    let mut bp = egui::pos2(mp.x+14.,mp.y-sz.y-6.);
                    bp.x = bp.x.clamp(canvas.left()+2.,canvas.right()-sz.x-2.);
                    bp.y = bp.y.clamp(canvas.top()+2.,canvas.bottom()-sz.y-2.);
                    let br = egui::Rect::from_min_size(bp,sz);
                    painter.rect_filled(br,3.,egui::Color32::from_rgba_premultiplied(0,0,0,210));
                    painter.rect_stroke(br,3.,egui::Stroke::new(1.,egui::Color32::from_rgba_premultiplied(200,200,200,100)),
                        egui::StrokeKind::Outside);
                    painter.galley(egui::pos2(bp.x+pad,bp.y+pad),g,egui::Color32::WHITE);
                }
            }
        });

        ui.ctx().request_repaint();
    }
}

// ---------------------------------------------------------------------------
// audio capture thread
// ---------------------------------------------------------------------------

fn run_audio(buf: Arc<Mutex<Vec<f32>>>) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let dev = host.default_input_device().ok_or("no mic")?;
    let cfg = dev.default_input_config()?;
    fn cb(b: &Arc<Mutex<Vec<f32>>>, data: &[f32]) {
        let mut buf = b.lock(); buf.extend_from_slice(data);
        let l = buf.len(); if l > FFT_SIZE*3 { buf.drain(0..l-FFT_SIZE*2); }
    }
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => { let b=Arc::clone(&buf);
            dev.build_input_stream::<f32,_,_>(cfg.into(),move|d,_|cb(&b,d),|e|eprintln!("audio:{e}"),None)? }
        cpal::SampleFormat::I16 => { let b=Arc::clone(&buf);
            dev.build_input_stream::<i16,_,_>(cfg.into(),
                move|d,_|{let v:Vec<f32>=d.iter().map(|&s|s as f32/i16::MAX as f32).collect();cb(&b,&v);},
                |e|eprintln!("audio:{e}"),None)? }
        cpal::SampleFormat::I32 => { let b=Arc::clone(&buf);
            dev.build_input_stream::<i32,_,_>(cfg.into(),
                move|d,_|{let v:Vec<f32>=d.iter().map(|&s|s as f32/i32::MAX as f32).collect();cb(&b,&v);},
                |e|eprintln!("audio:{e}"),None)? }
        cpal::SampleFormat::U16 => { let b=Arc::clone(&buf);
            dev.build_input_stream::<u16,_,_>(cfg.into(),
                move|d,_|{let v:Vec<f32>=d.iter().map(|&s|(s as f32-32768.)/32768.).collect();cb(&b,&v);},
                |e|eprintln!("audio:{e}"),None)? }
        _ => return Err(cpal::Error::new(cpal::ErrorKind::InvalidInput).into()),
    };
    stream.play()?; loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}

// ---------------------------------------------------------------------------
// drone output thread
// ---------------------------------------------------------------------------

fn run_drone(drone: Arc<Mutex<DroneState>>) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let dev = host.default_output_device().ok_or("no output")?;
    let cfg = dev.default_output_config()?;
    let sr = cfg.sample_rate() as f32;
    let stream = dev.build_output_stream::<f32,_,_>(cfg.into(),
        move|data:&mut[f32],_:&cpal::OutputCallbackInfo| {
            let mut d = drone.lock();
            if d.enabled && d.frequency > 0. {
                for s in data.iter_mut() { *s = (d.phase*2.*PI).sin()*d.amplitude;
                    d.phase = (d.phase + d.frequency/sr) % 1.; }
            } else { for s in data.iter_mut() { *s = 0.; }}
        }, |e| eprintln!("drone:{e}"), None)?;
    stream.play()?; loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), eframe::Error> {
    let sr = cpal::default_host().default_input_device()
        .and_then(|d| d.default_input_config().ok()).map(|c| c.sample_rate() as f32).unwrap_or(44100.);
    let audio = Arc::new(Mutex::new(Vec::new()));
    let drone = Arc::new(Mutex::new(DroneState::new()));

    std::thread::spawn({ let a=audio.clone(); move || { if let Err(e)=run_audio(a) { eprintln!("audio:{e}"); }}});
    std::thread::spawn({ let d=drone.clone(); move || { if let Err(e)=run_drone(d) { eprintln!("drone:{e}"); }}});

    eframe::run_native("Voice Harmonics Analyzer",
        eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1800.,860.]), ..Default::default() },
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(VoiceHarmApp::new(sr, audio, drone)))}))
}
