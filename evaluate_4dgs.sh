#!/bin/bash
# 4D Gaussian Splatting Evaluation Pipeline - Bash Wrapper
# Provides convenient CLI interface for common evaluation tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATE_SCRIPT="$SCRIPT_DIR/evaluate_4dgs.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
4D Gaussian Splatting Evaluation Pipeline

USAGE:
    ./evaluate_4dgs.sh [COMMAND] [OPTIONS]

COMMANDS:
    eval        Run evaluation on provided paths (default)
    setup       Validate environment setup
    generate    Generate test data
    help        Show this help message

EVAL OPTIONS:
    --render-path PATH      Path to rendered frames directory (required)
    --gt-path PATH          Path to ground truth frames directory (required)
    --mask-path PATH        Path to masks directory (optional)
    --device DEVICE         Device to use: cuda or cpu (default: auto)
    --output-dir DIR        Output directory for metrics (default: parent of renders)

SETUP OPTIONS:
    (none)

GENERATE OPTIONS:
    --output-dir DIR        Output directory (default: output)
    --experiment NAME       Experiment name (default: test_experiment)
    --num-frames N          Number of frames (default: 30)
    --width W               Image width (default: 512)
    --height H              Image height (default: 512)
    --image-type TYPE       Type: random, gradient, noise (default: random)
    --no-masks              Don't generate masks
    --no-offset             Don't add noise to renders

EXAMPLES:
    # Basic evaluation
    ./evaluate_4dgs.sh eval \\
        --render-path output/exp1/test/renders \\
        --gt-path output/exp1/test/gt

    # With masks
    ./evaluate_4dgs.sh eval \\
        --render-path output/exp1/test/renders \\
        --gt-path output/exp1/test/gt \\
        --mask-path output/exp1/test/masks

    # Generate test data
    ./evaluate_4dgs.sh generate --num-frames 50

    # Validate setup
    ./evaluate_4dgs.sh setup

EOF
}

# Validate environment
cmd_setup() {
    print_info "Validating evaluation environment..."
    python3 "$SCRIPT_DIR/setup_evaluation.py"
}

# Generate test data
cmd_generate() {
    local output_dir="output"
    local experiment="test_experiment"
    local num_frames="30"
    local width="512"
    local height="512"
    local image_type="random"
    local no_masks=""
    local no_offset=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir) output_dir="$2"; shift 2 ;;
            --experiment) experiment="$2"; shift 2 ;;
            --num-frames) num_frames="$2"; shift 2 ;;
            --width) width="$2"; shift 2 ;;
            --height) height="$2"; shift 2 ;;
            --image-type) image_type="$2"; shift 2 ;;
            --no-masks) no_masks="--no_masks"; shift ;;
            --no-offset) no_offset="--no_offset"; shift ;;
            *) print_error "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    print_info "Generating test data..."
    python3 "$SCRIPT_DIR/generate_test_data.py" \
        --output_dir "$output_dir" \
        --experiment_name "$experiment" \
        --num_frames "$num_frames" \
        --width "$width" \
        --height "$height" \
        --image_type "$image_type" \
        $no_masks \
        $no_offset
    
    print_success "Test data generated!"
}

# Run evaluation
cmd_eval() {
    local render_path=""
    local gt_path=""
    local mask_path=""
    local device="auto"
    local output_dir=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --render-path) render_path="$2"; shift 2 ;;
            --gt-path) gt_path="$2"; shift 2 ;;
            --mask-path) mask_path="$2"; shift 2 ;;
            --device) device="$2"; shift 2 ;;
            --output-dir) output_dir="$2"; shift 2 ;;
            *) print_error "Unknown option: $1"; exit 1 ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$render_path" ]]; then
        print_error "Missing required argument: --render-path"
        exit 1
    fi
    
    if [[ -z "$gt_path" ]]; then
        print_error "Missing required argument: --gt-path"
        exit 1
    fi
    
    # Check if directories exist
    if [[ ! -d "$render_path" ]]; then
        print_error "Render directory not found: $render_path"
        exit 1
    fi
    
    if [[ ! -d "$gt_path" ]]; then
        print_error "GT directory not found: $gt_path"
        exit 1
    fi
    
    if [[ -n "$mask_path" && ! -d "$mask_path" ]]; then
        print_warning "Mask directory not found, skipping BG-PSNR: $mask_path"
        mask_path=""
    fi
    
    # Build command
    local cmd="python3 '$EVALUATE_SCRIPT'"
    cmd="$cmd --render_path '$render_path'"
    cmd="$cmd --gt_path '$gt_path'"
    
    if [[ "$device" != "auto" ]]; then
        cmd="$cmd --device '$device'"
    fi
    
    if [[ -n "$mask_path" ]]; then
        cmd="$cmd --mask_path '$mask_path'"
    fi
    
    if [[ -n "$output_dir" ]]; then
        cmd="$cmd --output_dir '$output_dir'"
    fi
    
    print_info "Starting evaluation..."
    print_info "Renders: $render_path"
    print_info "GT: $gt_path"
    if [[ -n "$mask_path" ]]; then
        print_info "Masks: $mask_path"
    fi
    
    echo ""
    eval "$cmd"
    
    print_success "Evaluation complete!"
}

# Main
main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    case "$1" in
        eval)
            shift
            cmd_eval "$@"
            ;;
        setup)
            shift
            cmd_setup "$@"
            ;;
        generate)
            shift
            cmd_generate "$@"
            ;;
        help)
            usage
            ;;
        *)
            # Default to eval if first argument looks like an option
            if [[ "$1" == --* ]]; then
                cmd_eval "$@"
            else
                print_error "Unknown command: $1"
                usage
                exit 1
            fi
            ;;
    esac
}

main "$@"
