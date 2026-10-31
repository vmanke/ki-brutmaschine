//! A trained CNN and its file: the architecture, every trained value, and the running
//! statistics of its batch norms.
//!
//! The file is the JSON of `CnnModel`. It starts with `"kind":"cnn"`, which tells it from
//! the file of an MLP (`model::Model`). Unknown fields are refused, here and in the genome
//! parts. Reading a file allocates little for its values: the lists of trained values hold
//! at most `HARD_MAX_PARAMS` numbers in all, and the lists of running statistics at most
//! `MAX_RUNNING_VALUES`. These are the most any valid model has. `check` compares the file
//! with the network its architecture describes on a task, before any tensor is built.

use std::fmt;

use serde::de::{DeserializeSeed, Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::cnn::genome::{CnnGenome, MAX_CHANNELS, MAX_STAGES};
use crate::cnn::net::{self, Net, ParamSpec, Values};
use crate::cnn::ops::Tensor;
use crate::cnn::shape::Plan;
use crate::config::HARD_MAX_PARAMS;
use crate::genome::Activation;
use crate::task::TaskSpec;
use crate::util::{BrutError, Result};

/// The most running statistics a valid model has: a mean and a variance for each channel of
/// a batch norm in the stem and in every stage, at the widest width.
///
/// The stem and each stage have at most one norm today. A block with more norms (tickets 03
/// and 04) must raise this bound.
pub const MAX_RUNNING_VALUES: usize = 2 * (MAX_STAGES + 1) * MAX_CHANNELS;

/// The tag of a CNN file: the value `"cnn"` of its `"kind"` key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum CnnKind {
    #[default]
    #[serde(rename = "cnn")]
    Cnn,
}

/// A trained CNN.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CnnModel {
    /// Always `"cnn"`. It comes first in the JSON.
    kind: CnnKind,
    /// The stem, stages and head of the genome.
    pub arch: CnnGenome,
    /// The dense layers of the head (`Genome::hidden`).
    pub hidden: Vec<usize>,
    /// Their activation (`Genome::activation`).
    pub activation: Activation,
    /// The trained values in plan order, one list per tensor, each in row-major order. A
    /// convolution has its kernel (outputs, inputs / groups, k, k), or (outputs, inputs /
    /// groups, k) along a sequence, then its bias if it has one. A norm has its scale, then
    /// its shift. A dense layer has its matrix (outputs, inputs), then its bias.
    #[serde(deserialize_with = "read_params")]
    pub params: Vec<Vec<f32>>,
    /// For each batch norm in plan order: its running mean, then its running variance.
    #[serde(deserialize_with = "read_running")]
    pub batch_norm: Vec<Vec<f32>>,
}

impl CnnModel {
    /// A model with these parts and values; see the fields.
    pub fn new(
        arch: CnnGenome,
        hidden: Vec<usize>,
        activation: Activation,
        params: Vec<Vec<f32>>,
        batch_norm: Vec<Vec<f32>>,
    ) -> CnnModel {
        CnnModel {
            kind: CnnKind::Cnn,
            arch,
            hidden,
            activation,
            params,
            batch_norm,
        }
    }

    /// The number of trained values: the length of all lists in `params`. Saturates.
    pub fn param_count(&self) -> usize {
        self.params
            .iter()
            .fold(0usize, |acc, list| acc.saturating_add(list.len()))
    }

    /// True if every trained value and every running statistic is finite.
    pub fn is_finite(&self) -> bool {
        self.params
            .iter()
            .chain(&self.batch_norm)
            .flatten()
            .all(|v| v.is_finite())
    }

    /// The plan of this model on the task `spec`, if the model fits it: `plan`, then
    /// `check_values`. Neither builds a tensor.
    pub fn check(&self, spec: &TaskSpec) -> Result<Plan> {
        let plan = self.plan(spec)?;

        self.check_values(&plan)?;

        Ok(plan)
    }

    /// The layers of the architecture on the task `spec`: the shape check of its genome
    /// (`CnnGenome::plan`). It reads no trained value.
    pub fn plan(&self, spec: &TaskSpec) -> Result<Plan> {
        self.arch.plan(&self.hidden, self.activation, spec)
    }

    /// Check the values against `plan`, the plan of this model. `params` must hold one
    /// list per tensor of the plan, with exactly its number of values, and `batch_norm` two
    /// lists per batch norm, one value per channel each. Every value must be finite, and no
    /// running variance below 0.
    pub fn check_values(&self, plan: &Plan) -> Result<()> {
        let specs = net::param_specs(plan, self.activation);
        let channels = net::batch_norm_channels(plan);

        if self.params.len() != specs.len() {
            return Err(invalid(format!(
                "{} value lists, the plan has {} tensors",
                self.params.len(),
                specs.len()
            )));
        }

        if let Some(i) = (0..specs.len()).find(|&i| self.params[i].len() != specs[i].len()) {
            return Err(invalid(format!(
                "tensor {} has {} values, the plan needs {}",
                i + 1,
                self.params[i].len(),
                specs[i].len()
            )));
        }

        // The counts only, so that the message stays short whatever the file holds.
        let wanted: Vec<usize> = channels.iter().flat_map(|&c| [c, c]).collect();

        if self.batch_norm.len() != wanted.len() {
            return Err(invalid(format!(
                "{} lists of running statistics, the batch norms need {}",
                self.batch_norm.len(),
                wanted.len()
            )));
        }

        if let Some(i) = (0..wanted.len()).find(|&i| self.batch_norm[i].len() != wanted[i]) {
            return Err(invalid(format!(
                "list {} of running statistics has {} values, the batch norm needs {}",
                i + 1,
                self.batch_norm[i].len(),
                wanted[i]
            )));
        }

        if !self.is_finite() {
            return Err(invalid("a value is not finite".into()));
        }

        let negative = self
            .batch_norm
            .iter()
            .skip(1)
            .step_by(2)
            .flatten()
            .any(|&v| v < 0.0);

        if negative {
            return Err(invalid("a running variance is below 0".into()));
        }

        Ok(())
    }

    /// The network of this model on `spec`, after `check`.
    pub(crate) fn evaluator(&self, spec: &TaskSpec) -> Result<Evaluator> {
        let plan = self.check(spec)?;
        let specs = net::param_specs(&plan, self.activation);
        let running = self
            .batch_norm
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect();
        let values = Values::from_plan_order(&specs, &self.params, running);
        let net = Net::new(&plan, &specs, &values)?;
        let input = plan.input.channels.saturating_mul(plan.input.positions());
        let values_per_row = plan.steps.iter().fold(input, |acc, step| {
            acc.saturating_add(step.output.channels.saturating_mul(step.output.positions()))
        });

        Ok(Evaluator {
            net,
            values_per_row,
            outputs: spec.n_out,
        })
    }

    /// A model of `arch` with these values, from training (`cnn::train`).
    pub(crate) fn from_values(
        arch: CnnGenome,
        hidden: Vec<usize>,
        activation: Activation,
        specs: &[ParamSpec],
        values: &Values,
    ) -> CnnModel {
        let batch_norm = values
            .running
            .iter()
            .flat_map(|(mean, var)| [mean.clone(), var.clone()])
            .collect();

        CnnModel::new(
            arch,
            hidden,
            activation,
            values.plan_order(specs),
            batch_norm,
        )
    }
}

/// The forward pass of a checked model in evaluation mode, for scoring (`score::evaluate`).
pub(crate) struct Evaluator {
    net: Net,
    /// Values of one sample in the input (all channels) and after every layer.
    values_per_row: usize,
    outputs: usize,
}

impl Evaluator {
    /// Values of one sample in the input (all channels) and after every layer of the
    /// network. `score::evaluate` sizes its chunks of rows by it.
    ///
    /// It is not the memory of a row: the evaluation records no graph (`Net::forward_eval`),
    /// so it holds only a few layer outputs at a time, besides the buffers candle needs
    /// inside one operation. The scored bits of a CNN depend on the chunk sizes (see
    /// `score`), so a change of this count changes them.
    pub(crate) fn values_per_row(&self) -> usize {
        self.values_per_row
    }

    /// The raw outputs (logits for classification) of `rows` samples, row-major.
    pub(crate) fn outputs(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let mut shape = vec![rows];

        shape.extend_from_slice(self.net.sample());

        let out = self
            .net
            .forward_eval(&Tensor::from_f32(x.to_vec(), &shape)?)?;

        if out.dims() != [rows, self.outputs] {
            return Err(invalid(format!(
                "the network gives outputs of shape {:?}, not ({rows}, {})",
                out.dims(),
                self.outputs
            )));
        }

        out.to_vec_f32()
    }
}

/// An error of a model that does not fit its task.
fn invalid(what: String) -> BrutError {
    BrutError::Invalid(format!("cnn model: {what}"))
}

/// Reads the trained values: at most `HARD_MAX_PARAMS` values in all, and no empty list.
///
/// Every tensor of a network holds a value, and no valid model holds more values than
/// `HARD_MAX_PARAMS`. So the reader stops a hostile file before it allocates more.
fn read_params<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<Vec<f32>>, D::Error> {
    deserializer.deserialize_seq(Lists {
        cap: HARD_MAX_PARAMS,
    })
}

/// Reads the running statistics: at most `MAX_RUNNING_VALUES` values in all, and no empty
/// list.
fn read_running<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<Vec<f32>>, D::Error> {
    deserializer.deserialize_seq(Lists {
        cap: MAX_RUNNING_VALUES,
    })
}

/// The visitor of `read_params` and `read_running`: lists of at most `cap` values in all.
struct Lists {
    cap: usize,
}

impl<'de> Visitor<'de> for Lists {
    type Value = Vec<Vec<f32>>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "lists of at most {} numbers in all", self.cap)
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut left = self.cap;
        let mut lists = Vec::new();

        while let Some(list) = seq.next_element_seed(List {
            cap: self.cap,
            left: &mut left,
        })? {
            lists.push(list);
        }

        Ok(lists)
    }
}

/// One list of `Lists`; `left` counts the values still allowed of `cap`.
struct List<'a> {
    cap: usize,
    left: &'a mut usize,
}

impl<'de> DeserializeSeed<'de> for List<'_> {
    type Value = Vec<f32>;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Vec<f32>, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for List<'_> {
    type Value = Vec<f32>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a list of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<f32>, A::Error> {
        let mut values = Vec::new();

        while let Some(value) = seq.next_element::<f32>()? {
            if *self.left == 0 {
                return Err(A::Error::custom(format!("more than {} values", self.cap)));
            }

            *self.left -= 1;
            values.push(value);
        }

        if values.is_empty() {
            return Err(A::Error::custom("an empty list of values"));
        }

        Ok(values)
    }
}

/// Models for the tests of the crate.
#[cfg(test)]
pub(crate) mod testing {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::genome::Genome;

    /// The untrained model of `genome` on `spec`, with values drawn from `seed`.
    pub(crate) fn untrained(genome: &Genome, spec: &TaskSpec, seed: u64) -> CnnModel {
        let plan = genome.cnn_plan(spec).expect("plan");
        let specs = net::param_specs(&plan, genome.activation);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let values = Values::initial(
            &specs,
            &net::batch_norm_channels(&plan),
            genome.init_scale,
            &mut rng,
        );
        let crate::genome::Arch::Cnn(cnn) = &genome.arch else {
            panic!("expected a CNN genome");
        };

        CnnModel::from_values(
            cnn.clone(),
            genome.hidden.clone(),
            genome.activation,
            &specs,
            &values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::testing::untrained;
    use super::*;
    use crate::cnn::genome::{
        CnnActivation, Dims, Downsample, Head, Norm, Stage, Stem, testing,
    };
    use crate::task::TaskKind;

    /// The VGG-like test genome on the shapes task: batch norm, max pooling, flatten.
    fn vgg() -> (CnnModel, TaskSpec) {
        let spec = TaskKind::Shapes.spec();

        (untrained(&testing::vgg(), &spec, 1), spec)
    }

    #[test]
    fn a_model_file_starts_with_its_kind_and_parses_back_equal() {
        let (model, spec) = vgg();
        let text = serde_json::to_string(&model).expect("json");

        assert!(text.starts_with("{\"kind\":\"cnn\",\"arch\":{\"dims\":\"TwoD\""));
        assert_eq!(
            serde_json::from_str::<CnnModel>(&text).expect("parse"),
            model
        );
        assert!(model.check(&spec).is_ok());
        assert_eq!(model.param_count(), testing::vgg().param_count(&spec));
    }

    #[test]
    fn check_refuses_values_that_do_not_fit_the_plan() {
        let (model, spec) = vgg();
        let refused = |change: &dyn Fn(&mut CnnModel), what: &str| {
            let mut bad = model.clone();

            change(&mut bad);

            let message = bad.check(&spec).expect_err(what).to_string();

            assert!(message.contains("cnn"), "{what}: {message}");
        };

        refused(
            &|m| {
                m.params.pop();
            },
            "a tensor is missing",
        );
        refused(&|m| m.params.push(vec![0.0]), "one tensor too many");
        refused(&|m| m.params[0].push(0.0), "one value too many");
        refused(
            &|m| {
                m.params[3].pop();
            },
            "one value too few",
        );
        refused(&|m| m.params[2][0] = f32::NAN, "a NaN");
        refused(&|m| m.params[5][1] = f32::INFINITY, "an infinity");
        refused(&|m| m.batch_norm[1][0] = -1.0, "a negative variance");
        refused(
            &|m| {
                m.batch_norm.pop();
            },
            "a running variance is missing",
        );
        refused(&|m| m.batch_norm[0].push(0.0), "a running mean too long");
        refused(
            &|m| m.hidden = vec![4],
            "dense layers the values do not have",
        );
        refused(
            &|m| m.arch.stages[0].width = 9,
            "a width the values do not have",
        );
        refused(
            &|m| m.arch.dims = crate::cnn::genome::Dims::OneD,
            "a 1-D genome on an image",
        );
        refused(&|m| m.arch.stem.kernel = 4, "an even kernel");

        // A sequence task does not fit the image model at all.
        assert!(model.check(&TaskKind::Motif1d.spec()).is_err());
    }

    #[test]
    fn unknown_fields_are_refused_also_inside_the_genome() {
        let (model, _) = vgg();
        let text = serde_json::to_string(&model).expect("json");

        for bad in [
            text.replacen("{\"kind\":\"cnn\",", "{\"kind\":\"cnn\",\"extra\":1,", 1),
            text.replacen("\"stem\":{", "\"stem\":{\"extra\":1,", 1),
            text.replacen("\"kind\":\"cnn\"", "\"kind\":\"mlp\"", 1),
            text.replacen("\"kind\":\"cnn\",", "", 1),
        ] {
            assert!(
                serde_json::from_str::<CnnModel>(&bad).is_err(),
                "{}",
                &bad[..60]
            );
        }
    }

    /// A model file whose lists hold these many values.
    fn file_with_lists(lengths: &[usize]) -> String {
        let (model, _) = vgg();
        let text = serde_json::to_string(&CnnModel {
            params: Vec::new(),
            ..model
        })
        .expect("json");
        let lists: Vec<String> = lengths
            .iter()
            .map(|&n| format!("[{}]", vec!["0"; n].join(",")))
            .collect();

        text.replacen(
            "\"params\":[]",
            &format!("\"params\":[{}]", lists.join(",")),
            1,
        )
    }

    #[test]
    fn the_reader_stops_at_the_value_cap_and_refuses_empty_lists() {
        let fits = file_with_lists(&[HARD_MAX_PARAMS / 2, HARD_MAX_PARAMS / 2]);

        assert!(serde_json::from_str::<CnnModel>(&fits).is_ok());

        for lengths in [
            vec![HARD_MAX_PARAMS + 1],
            vec![HARD_MAX_PARAMS / 2, HARD_MAX_PARAMS / 2, 1],
        ] {
            let message = serde_json::from_str::<CnnModel>(&file_with_lists(&lengths))
                .expect_err("over the cap")
                .to_string();

            assert!(
                message.contains(&format!("more than {HARD_MAX_PARAMS} values")),
                "{message}"
            );
        }

        let message = serde_json::from_str::<CnnModel>(&file_with_lists(&[3, 0, 2]))
            .expect_err("an empty list")
            .to_string();

        assert!(message.contains("an empty list"), "{message}");
    }

    /// A model file whose running statistics are these many lists of one value each.
    fn file_with_running_lists(lists: usize) -> String {
        let (model, _) = vgg();

        serde_json::to_string(&CnnModel {
            batch_norm: vec![vec![1.0]; lists],
            ..model
        })
        .expect("json")
    }

    #[test]
    fn the_running_statistics_have_their_own_small_cap() {
        // The widest layout: the stem and every stage with a batch norm at the widest width.
        let wide = Stage {
            norm: Norm::Batch,
            ..testing::plain(MAX_CHANNELS, 1, CnnActivation::Relu)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: MAX_CHANNELS,
                kernel: 1,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages: vec![wide; MAX_STAGES],
            head: Head::GlobalAvgPool,
        };
        let genome = testing::genome(cnn, Vec::new(), Activation::Relu);
        let plan = genome.cnn_plan(&TaskKind::Shapes.spec()).expect("plan");
        let running: usize = net::batch_norm_channels(&plan).iter().map(|c| 2 * c).sum();

        assert_eq!(running, MAX_RUNNING_VALUES);

        // The reader takes that many values, and stops at one more.
        assert!(serde_json::from_str::<CnnModel>(&file_with_running_lists(MAX_RUNNING_VALUES)).is_ok());

        let message = serde_json::from_str::<CnnModel>(&file_with_running_lists(
            MAX_RUNNING_VALUES + 1,
        ))
        .expect_err("over the cap")
        .to_string();

        assert!(
            message.contains(&format!("more than {MAX_RUNNING_VALUES} values")),
            "{message}"
        );
    }

    #[test]
    fn a_refusal_of_many_lists_gives_a_short_message() {
        let (model, spec) = vgg();
        let many: CnnModel =
            serde_json::from_str(&file_with_running_lists(MAX_RUNNING_VALUES)).expect("parse");
        let message = many.check(&spec).expect_err("too many lists").to_string();

        assert!(
            message.contains(&format!("{MAX_RUNNING_VALUES} lists of running statistics")),
            "{message}"
        );
        assert!(message.len() < 120, "{message}");

        let mut long = model.clone();

        long.batch_norm[2] = vec![1.0; 1000];

        let message = long.check(&spec).expect_err("a long list").to_string();

        assert!(message.contains("list 3 of running statistics has 1000 values"), "{message}");
        assert!(message.len() < 120, "{message}");
    }

    #[test]
    fn evaluation_uses_the_running_statistics_of_batch_norm() {
        let (mut model, spec) = vgg();
        let rows = 3;
        let x: Vec<f32> = (0..rows * spec.n_in)
            .map(|i| (i % 7) as f32 / 7.0)
            .collect();
        let outputs = |m: &CnnModel, rows: usize| {
            m.evaluator(&spec)
                .expect("evaluator")
                .outputs(&x[..rows * spec.n_in], rows)
                .expect("outputs")
        };
        let three = outputs(&model, rows);

        // Each row is scored on its own: the batch does not change its outputs.
        assert_eq!(&three[..spec.n_out], &outputs(&model, 1)[..]);

        // The running statistics are used: other ones give other outputs.
        model.batch_norm[0] = vec![0.5; model.batch_norm[0].len()];
        model.batch_norm[1] = vec![4.0; model.batch_norm[1].len()];

        assert_ne!(three, outputs(&model, rows));
    }
}
