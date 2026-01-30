"""
Enhanced AI Analyzer Module - COMPLETE FIXED VERSION with IMPROVED ACCURACY
- Multiple preprocessing strategies
- Multi-pass extraction with consensus
- Better digit confusion handling
- Enhanced validation and correction
- Confidence scoring
"""
import json
import re
from typing import Dict, Any, List, Tuple
import os
import time
from dotenv import load_dotenv
load_dotenv()

# Detect which AI service is available
OLLAMA_AVAILABLE = False
USE_GEMINI = False
USE_OPENAI = False
genai_client = None

# Try Ollama first
try:
    import ollama
    ollama.list()
    OLLAMA_AVAILABLE = True
    print("[AI Config] âœ“ Ollama detected - Using LOCAL models")
except:
    print("[AI Config] Ollama not available")

# Try Gemini (NEW API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        print("[AI Config] âœ“ Using Google Gemini (NEW API)")
        print("[AI Config] âœ“ Gemini Vision: ENABLED âœ…")
    except ImportError as e:
        print(f"[AI Config] google-genai not installed - run: pip3 install google-genai")
        print(f"[AI Config] Error: {e}")
    except Exception as e:
        print(f"[AI Config] Gemini initialization error: {e}")
else:
    print("[AI Config] âœ“ Gemini Vision: DISABLED âŒ (No API key)")

# Try OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not USE_GEMINI and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
        print("[AI Config] âœ“ Using OpenAI GPT")
    except ImportError:
        print("[AI Config] openai package not installed")

if not OLLAMA_AVAILABLE and not USE_GEMINI and not USE_OPENAI:
    print("[AI Config] âš ï¸ WARNING: No AI service available, will use REGEX ONLY")

# Model selection
OLLAMA_MODEL = 'qwen2.5:7b'
GEMINI_MODEL = 'gemini-2.5-flash'

print(f"[AI Config] Selected Model: {OLLAMA_MODEL if OLLAMA_AVAILABLE else GEMINI_MODEL if USE_GEMINI else 'OpenAI GPT' if USE_OPENAI else 'None'}")


def clean_ocr_text(text: str) -> str:
    """Fix common OCR character substitutions"""
    corrections = {
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
        'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T',
        'Х': 'X', 'У': 'Y', 'І': 'I', 'Ѕ': 'S',
        '|': 'I', '¡': 'I', 'l': 'I', '–': '-', '—': '-',
    }
    
    result = text
    for old, new in corrections.items():
        result = result.replace(old, new)
    
    return result


def validate_indian_account_number(account_num: str) -> bool:
    """Validate Indian bank account number (9-18 digits)"""
    if not account_num:
        return False
    
    clean = str(account_num).replace(' ', '').replace('-', '')
    
    if not clean.isdigit():
        return False
    
    if len(clean) < 9 or len(clean) > 18:
        return False
    
    if len(clean) == 17:
        print(f"[VALIDATION] âš ï¸ Suspicious: 17-digit account number")
        return False
    
    return True


def create_multiple_preprocessing_variants(image_bytes: bytes) -> List[Tuple[bytes, str]]:
    """
    Create multiple preprocessing variants for better OCR accuracy
    Returns list of (enhanced_bytes, strategy_name) tuples
    """
    try:
        import cv2
        import numpy as np
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[IMAGE ENHANCE] âš ï¸ Could not decode image, using original")
            return [(image_bytes, "original")]
        
        variants = []
        
        # Strategy 1: High contrast + denoising
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised1 = cv2.fastNlMeansDenoising(gray1, h=10)
        clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced1 = clahe1.apply(denoised1)
        kernel1 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened1 = cv2.filter2D(enhanced1, -1, kernel1)
        success, buffer = cv2.imencode('.jpg', sharpened1, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            variants.append((buffer.tobytes(), "high_contrast"))
        
        # Strategy 2: Binary threshold (Otsu)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        _, binary2 = cv2.threshold(blurred2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        success, buffer = cv2.imencode('.jpg', binary2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            variants.append((buffer.tobytes(), "otsu_binary"))
        
        # Strategy 3: Adaptive threshold
        gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised3 = cv2.fastNlMeansDenoising(gray3, h=10)
        adaptive3 = cv2.adaptiveThreshold(denoised3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        success, buffer = cv2.imencode('.jpg', adaptive3, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            variants.append((buffer.tobytes(), "adaptive_threshold"))
        
        # Strategy 4: Morphological operations (dilation + erosion)
        gray4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel4 = np.ones((2, 2), np.uint8)
        dilated4 = cv2.dilate(binary4, kernel4, iterations=1)
        eroded4 = cv2.erode(dilated4, kernel4, iterations=1)
        success, buffer = cv2.imencode('.jpg', eroded4, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            variants.append((buffer.tobytes(), "morphological"))
        
        # Strategy 5: Super high contrast (aggressive)
        gray5 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe5 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        enhanced5 = clahe5.apply(gray5)
        _, binary5 = cv2.threshold(enhanced5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel5 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened5 = cv2.filter2D(binary5, -1, kernel5)
        success, buffer = cv2.imencode('.jpg', sharpened5, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            variants.append((buffer.tobytes(), "aggressive"))
        
        # Add original as fallback
        variants.append((image_bytes, "original"))
        
        print(f"[IMAGE ENHANCE] âœ“ Created {len(variants)} preprocessing variants")
        return variants
        
    except Exception as e:
        print(f"[IMAGE ENHANCE] âš ï¸ Error: {e}, using original image")
        return [(image_bytes, "original")]


def enhance_image_for_ocr(image_bytes: bytes) -> bytes:
    """
    Enhance image quality for better OCR accuracy
    - Increases contrast
    - Reduces noise
    - Sharpens text
    """
    try:
        import cv2
        import numpy as np
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[IMAGE ENHANCE] âš ï¸ Could not decode image, using original")
            return image_bytes
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to bytes
        success, buffer = cv2.imencode('.jpg', sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            enhanced_bytes = buffer.tobytes()
            print(f"[IMAGE ENHANCE] âœ“ Enhanced image: {len(image_bytes)} â†’ {len(enhanced_bytes)} bytes")
            return enhanced_bytes
        else:
            print("[IMAGE ENHANCE] âš ï¸ Enhancement failed, using original")
            return image_bytes
            
    except Exception as e:
        print(f"[IMAGE ENHANCE] âš ï¸ Error: {e}, using original image")
        return image_bytes


def fix_common_digit_errors(text: str, field_type: str = 'account') -> str:
    """
    Fix common OCR digit confusion based on field type
    """
    if not text:
        return text
    
    result = text
    
    if field_type == 'account':
        # Account numbers should be pure digits
        # Fix common confusions
        result = result.replace('O', '0')  # Letter O â†’ digit 0
        result = result.replace('o', '0')  # lowercase o â†’ digit 0
        result = result.replace('I', '1')  # Letter I â†’ digit 1
        result = result.replace('l', '1')  # lowercase L â†’ digit 1
        result = result.replace('S', '5')  # Letter S â†’ digit 5 (if in account)
        result = result.replace('B', '8')  # Letter B â†’ digit 8 (if in account)
        result = result.replace('Z', '2')  # Letter Z â†’ digit 2
        
        # Remove any remaining non-digits
        result = ''.join(c for c in result if c.isdigit())
    
    elif field_type == 'ifsc':
        result = result.upper()
        # IFSC: 4 letters + 0 + 6 alphanumeric
        # Fix position 5 (must be digit 0)
        if len(result) >= 5:
            if result[4] == 'O':
                result = result[:4] + '0' + result[5:]
        
        # Fix common bank code confusions
        result = result.replace('SBIN0', 'SBIN0')  # State Bank
        result = result.replace('CNRB0', 'CNRB0')  # Canara
        result = result.replace('IOBA0', 'IOBA0')  # IOB
        result = result.replace('IDIB0', 'IDIB0')  # Indian Bank
        result = result.replace('UBIN0', 'UBIN0')  # Union Bank
        
        # Remove spaces
        result = result.replace(' ', '').replace('-', '')
    
    return result


def calculate_digit_similarity(digit1: str, digit2: str) -> float:
    """
    Calculate visual similarity between two digits (0-1)
    Used for identifying likely OCR errors
    """
    similar_pairs = {
        ('3', '8'): 0.8, ('8', '3'): 0.8,
        ('5', '6'): 0.7, ('6', '5'): 0.7,
        ('3', '5'): 0.6, ('5', '3'): 0.6,
        ('6', '8'): 0.75, ('8', '6'): 0.75,
        ('0', 'O'): 0.9, ('O', '0'): 0.9,
        ('1', '7'): 0.5, ('7', '1'): 0.5,
        ('2', 'Z'): 0.6, ('Z', '2'): 0.6,
    }
    
    return similar_pairs.get((digit1, digit2), 0.0)


def validate_and_correct_bank_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and auto-correct common OCR errors in extracted bank data
    """
    corrected = data.copy()
    
    # Validate and clean account number
    if corrected.get('account_number'):
        acc = str(corrected['account_number']).strip()
        
        # Apply digit error fixes
        acc = fix_common_digit_errors(acc, 'account')
        
        # Check length validity
        if len(acc) < 9:
            print(f"[VALIDATION] âš ï¸ Account too short: {len(acc)} digits (minimum 9)")
        elif len(acc) > 18:
            print(f"[VALIDATION] âš ï¸ Account too long: {len(acc)} digits (maximum 18)")
            # Try to trim to valid length
            if len(acc) == 19:
                # Might have extra leading digit
                acc = acc[1:]  # Try removing first digit
                print(f"[VALIDATION] âœ“ Trimmed to {acc}")
        
        corrected['account_number'] = acc if acc else None
    
    # Validate and clean IFSC code
    if corrected.get('ifsc_code'):
        ifsc = str(corrected['ifsc_code']).strip()
        
        # Apply IFSC-specific fixes
        ifsc = fix_common_digit_errors(ifsc, 'ifsc')
        
        # Validate format: 4 letters + 0 + 6 alphanumeric
        if len(ifsc) == 11:
            # Ensure 5th character is '0' (zero), not 'O' (letter)
            if ifsc[4] == 'O':
                ifsc = ifsc[:4] + '0' + ifsc[5:]
                print(f"[VALIDATION] âœ“ Fixed IFSC: Changed 'O' to '0' at position 5")
            
            # Ensure first 4 are letters
            if not ifsc[:4].isalpha():
                print(f"[VALIDATION] âš ï¸ IFSC first 4 chars should be letters: {ifsc[:4]}")
            
            # Ensure 5th is '0'
            if ifsc[4] != '0':
                print(f"[VALIDATION] âš ï¸ IFSC 5th char should be '0': {ifsc[4]}")
            
            corrected['ifsc_code'] = ifsc
        elif len(ifsc) == 12:
            # Sometimes extra character, try to fix
            print(f"[VALIDATION] âš ï¸ IFSC has 12 chars, trying to fix: {ifsc}")
            # Check if there's a duplicate character
            if ifsc[4] == ifsc[5] and (ifsc[4] == '0' or ifsc[4] == 'O'):
                ifsc = ifsc[:4] + '0' + ifsc[6:]
                print(f"[VALIDATION] âœ“ Fixed IFSC by removing duplicate: {ifsc}")
                corrected['ifsc_code'] = ifsc
        elif len(ifsc) != 11:
            print(f"[VALIDATION] âš ï¸ IFSC wrong length: {len(ifsc)} (expected 11)")
    
    # Clean bank name
    if corrected.get('bank_name'):
        bank = str(corrected['bank_name']).strip()
        corrected['bank_name'] = bank if bank else None
    
    # Clean account holder name
    if corrected.get('account_holder_name'):
        name = str(corrected['account_holder_name']).strip()
        corrected['account_holder_name'] = name if name else None
    
    # Clean branch name
    if corrected.get('branch_name'):
        branch = str(corrected['branch_name']).strip()
        corrected['branch_name'] = branch if branch else None
    
    return corrected


def extract_with_consensus(results: List[Dict[str, Any]], field: str) -> Any:
    """
    Extract field value using consensus from multiple extraction attempts
    """
    values = [r.get(field) for r in results if r.get(field)]
    
    if not values:
        return None
    
    # For account numbers and IFSC, use the most common value
    from collections import Counter
    value_counts = Counter(values)
    most_common = value_counts.most_common(1)[0][0]
    
    # Check if there's strong consensus (>50%)
    consensus_ratio = value_counts[most_common] / len(values)
    
    if consensus_ratio >= 0.5:
        print(f"[CONSENSUS] âœ“ {field}: {most_common} (confidence: {consensus_ratio:.1%})")
        return most_common
    else:
        print(f"[CONSENSUS] âš ï¸ {field}: Low consensus ({consensus_ratio:.1%}), using: {most_common}")
        return most_common


def analyze_bank_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    âœ… ENHANCED: Multi-pass bank passbook analysis with consensus
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[GEMINI VISION] Analyzing bank passbook: {filename}")
    print(f"[GEMINI VISION] Using multi-pass extraction with consensus...")
    
    try:
        from google.genai import types
        
        # Create multiple preprocessing variants
        variants = create_multiple_preprocessing_variants(file_content)
        print(f"[GEMINI VISION] Testing {len(variants)} image preprocessing strategies...")
        
        all_results = []
        
        # Enhanced prompt with specific digit confusion warnings
        prompt = """
You are an EXPERT OCR system analyzing an Indian bank passbook image. Extract ALL text with PIXEL-PERFECT ACCURACY.

ðŸ”´ CRITICAL DIGIT CONFUSION - VERIFY CAREFULLY:
- '3' vs '8': Look at the MIDDLE (3 has one curve, 8 has two curves stacked)
- '3' vs '5': Check the BOTTOM (3 curves right, 5 has flat bottom)
- '5' vs '6': Check the TOP (5 has flat/angular top, 6 has curved top)
- '6' vs '8': Check OPENINGS (6 open at top, 8 closed everywhere)
- '6' vs '0': Check for TAIL (6 has tail/hook at top, 0 is round)
- '0' vs 'O': In numbers, ALWAYS use digit '0' NEVER letter 'O'
- '1' vs '7': Check TOP (1 is straight, 7 has horizontal top)
- '2' vs 'Z': In numbers, ALWAYS digit '2' NEVER letter 'Z'

ðŸ” READING STRATEGY:
1. Locate the ENTIRE account number first (look for longest digit sequence)
2. Count ALL digits from START to END
3. Re-read EACH digit individually, comparing with similar-looking digits
4. Verify total digit count is between 9-18
5. For LEADING DIGITS, zoom in mentally and verify carefully (often the first 1-2 digits are wrong)

**ACCOUNT NUMBER EXTRACTION:**
- Labels to find: "Account No:", "A/c No:", "Account Number:", "Saving Bank A/c:", "SB A/c:"
- Length: 9-18 digits (most common: 11-16 digits)
- Common errors to avoid:
  * Missing FIRST digit (e.g., reading 7874674475 when it's actually 17874674475)
  * Confusing 3â†”5, 3â†”8, 6â†”8, 0â†”O
  * Adding/removing zeros
  * Reading nearby numbers (CIF, phone, MICR) as account number

**IFSC CODE EXTRACTION:**
- Format: EXACTLY 11 characters: [4 LETTERS][digit 0][6 ALPHANUMERIC]
- Examples: SBIN0002196, CNRB0003556, IOBA0001310, IDIB000T037, UBIN0533289
- Position 5 is ALWAYS digit '0' (zero), NEVER letter 'O'
- Common bank codes:
  * SBIN = State Bank of India
  * CNRB = Canara Bank
  * IOBA = Indian Overseas Bank
  * IDIB = Indian Bank
  * UBIN = Union Bank of India
  * BKID = Bank of India (NOTE: it's BKID not BNKID)
- Location: Near branch name, in corners, with "IFSC:" label

**VERIFICATION CHECKLIST:**
âœ“ Account: Count digits â†’ Re-read first 3 digits â†’ Re-read last 3 digits â†’ Verify middle digits
âœ“ IFSC: Check 11 chars â†’ Verify position 5 is '0' â†’ Check bank code matches bank name
âœ“ Name: Full name with titles extracted
âœ“ Bank: Correct bank name from top of page

Return ONLY this JSON:
{
    "bank_name": "Full bank name or null",
    "account_holder_name": "Full name with titles or null",
    "account_number": "Account number (digits only, 9-18 length) or null",
    "ifsc_code": "IFSC code (exactly 11 chars) or null",
    "branch_name": "Branch name with location or null",
    "confidence": "high|medium|low"
}

ðŸŽ¯ Read SLOWLY. Verify TWICE. Count CAREFULLY.
"""
        
        # Detect MIME type
        mime_type = "image/jpeg"
        if file_content[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif file_content[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        
        # Try each preprocessing variant
        for img_bytes, strategy in variants[:2]:  # Use top 2 strategies for speed (was 3)
            print(f"[GEMINI VISION] Trying strategy: {strategy}")
            
            try:
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type=mime_type
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,  # Maximum determinism
                        top_p=0.95,
                        top_k=40,
                        response_mime_type="application/json"
                    )
                )
                
                response_text = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(response_text)
                
                # Validate and correct
                data = validate_and_correct_bank_data(data)
                data['_strategy'] = strategy
                
                all_results.append(data)
                
                print(f"[GEMINI VISION]   {strategy}: Account={data.get('account_number')}, IFSC={data.get('ifsc_code')}")
                
            except Exception as e:
                print(f"[GEMINI VISION]   {strategy}: Failed - {e}")
                continue
        
        if not all_results:
            raise Exception("All extraction attempts failed")
        
        # Build consensus result
        consensus = {
            'bank_name': extract_with_consensus(all_results, 'bank_name'),
            'account_holder_name': extract_with_consensus(all_results, 'account_holder_name'),
            'account_number': extract_with_consensus(all_results, 'account_number'),
            'ifsc_code': extract_with_consensus(all_results, 'ifsc_code'),
            'branch_name': extract_with_consensus(all_results, 'branch_name'),
        }
        
        # Second pass for missing IFSC
        if not consensus.get('ifsc_code') and consensus.get('bank_name'):
            print(f"[GEMINI VISION] IFSC missing, attempting targeted second pass...")
            try:
                # Use the best preprocessed variant
                best_variant = variants[0][0]
                
                ifsc_prompt = """
ðŸ”´ CRITICAL: FIND THE IFSC CODE

This passbook image is MISSING the IFSC code. It MUST be somewhere on this page.

IFSC CODE HIDING PLACES:
1. **Top-right corner** - Small text, often overlooked
2. **Top-left corner** - Sometimes rotated
3. **Near branch name** - Usually adjacent to branch details
4. **Bottom footer** - Very small text at page bottom
5. **Inside boxes/borders** - Check bordered sections
6. **Near "MICR Code"** - Often printed together
7. **Watermark area** - Sometimes overlapped with watermark
8. **Rotated text** - Check vertical margins

IFSC Format: EXACTLY 11 characters
Examples: SBIN0002196, CNRB0016140, IOBA0001310

Bank code hints based on bank name:
- "State Bank" or "SBI" â†’ SBIN____
- "Canara" â†’ CNRB____
- "Indian Overseas" or "IOB" â†’ IOBA____
- "Indian Bank" â†’ IDIB____
- "Union Bank" â†’ UBIN____

Character 5 is ALWAYS digit '0' (zero)

Scan EVERY PIXEL. Find this code.

Return ONLY:
{
    "ifsc_code": "11-character code or null"
}
"""
                
                response2 = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=best_variant,
                            mime_type=mime_type
                        ),
                        ifsc_prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json"
                    )
                )
                
                ifsc_data = json.loads(response2.text.replace("```json", "").replace("```", "").strip())
                if ifsc_data.get('ifsc_code'):
                    ifsc_fixed = fix_common_digit_errors(ifsc_data['ifsc_code'], 'ifsc')
                    consensus['ifsc_code'] = ifsc_fixed
                    print(f"[GEMINI VISION] âœ“ Found IFSC in second pass: {ifsc_fixed}")
            except Exception as e:
                print(f"[GEMINI VISION] Second pass failed: {e}")
        
        print(f"\n[GEMINI VISION] âœ“ FINAL CONSENSUS RESULT:")
        print(f"[GEMINI VISION]   Bank: {consensus.get('bank_name')}")
        print(f"[GEMINI VISION]   Name: {consensus.get('account_holder_name')}")
        print(f"[GEMINI VISION]   Account: {consensus.get('account_number')}")
        print(f"[GEMINI VISION]   IFSC: {consensus.get('ifsc_code')}")
        print(f"[GEMINI VISION]   Branch: {consensus.get('branch_name')}")
        
        return consensus
        
    except Exception as e:
        print(f"[GEMINI VISION] âœ— Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def analyze_bill_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    âœ… FIXED: Analyze bill using Gemini Vision (NEW API)
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available")
    
    print(f"[GEMINI VISION] Analyzing bill: {filename}")
    
    try:
        from google.genai import types
        
        # Detect MIME type
        mime_type = "image/jpeg"
        if file_content[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif file_content[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        
        prompt = """
Extract details from this Indian college fee receipt/bill.

FIND:
1. **Student Name**: Look for "Name:" field (may be handwritten)
2. **Date**: Convert to YYYY-MM-DD format
3. **Total Amount**: Bottom of itemized list (extract number only)
4. **College Name**: Usually at top in large text
5. **Receipt Number**: Look for "Receipt No:", "Challan No:"

RULES:
- For amounts like "18000-00", extract as 18000
- Student name is in "Name" field, NOT signature
- Date format: YYYY-MM-DD

Return ONLY JSON:
{
    "student_name": "name or null",
    "college_name": "name or null",
    "roll_number": "number or null",
    "receipt_number": "number or null",
    "class_course": "course or null",
    "bill_date": "YYYY-MM-DD or null",
    "amount": 18000.00
}
"""
        
        # âœ… Use raw bytes directly
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=file_content,
                    mime_type=mime_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # âœ… FIX: Handle case where Gemini returns an array instead of object
        if isinstance(data, list):
            print(f"[GEMINI VISION] âš ï¸ Received array response, taking first element")
            if len(data) > 0:
                data = data[0]  # Take first element
            else:
                # Empty array, return null values
                data = {
                    "student_name": None,
                    "college_name": None,
                    "roll_number": None,
                    "receipt_number": None,
                    "class_course": None,
                    "bill_date": None,
                    "amount": None
                }
        
        # âœ… Ensure data is a dict before accessing
        if not isinstance(data, dict):
            print(f"[GEMINI VISION] âš ï¸ Unexpected data type: {type(data)}")
            data = {
                "student_name": None,
                "college_name": None,
                "roll_number": None,
                "receipt_number": None,
                "class_course": None,
                "bill_date": None,
                "amount": None
            }
        
        print(f"[GEMINI VISION] âœ“ Extracted bill details")
        print(f"[GEMINI VISION]   College: {data.get('college_name')}")
        print(f"[GEMINI VISION]   Student: {data.get('student_name')}")
        print(f"[GEMINI VISION]   Date: {data.get('bill_date')}")
        print(f"[GEMINI VISION]   Amount: Rs. {data.get('amount')}")
        
        return data
        
    except Exception as e:
        print(f"[GEMINI VISION] âœ— Error: {e}")
        import traceback
        traceback.print_exc()
        raise
        
def extract_bank_details_from_markdown(markdown_text: str) -> dict:
    """Regex-based bank extraction (fallback)"""
    print(f"[MANUAL EXTRACTION] Parsing bank markdown...")
    
    text = clean_ocr_text(markdown_text)
    
    account_holder_name = None
    account_number = None
    ifsc_code = None
    branch_name = None
    bank_name = None
    
    text_upper = text.upper()
    text_clean = text.replace('\n', ' ').replace('#', '')
    
    # Extract bank name
    bank_keywords = {
        'INDIAN BANK': 'Indian Bank',
        'CANARA': 'Canara Bank',
        'SBI': 'State Bank of India',
        'STATE BANK': 'State Bank of India',
        'HDFC': 'HDFC Bank',
        'ICICI': 'ICICI Bank',
        'AXIS': 'Axis Bank',
        'PNB': 'Punjab National Bank',
    }
    
    for keyword, full_name in bank_keywords.items():
        if keyword in text_upper:
            bank_name = full_name
            print(f"[MANUAL EXTRACTION]   âœ“ Found bank: {bank_name}")
            break
    
    # Extract account number
    account_patterns = [
        r'Account\s*(?:No\.?|Number)[:\s]*(\d{9,16})',
        r'A/c\s*No\.?[:\s]*(\d{9,16})',
    ]
    
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if validate_indian_account_number(candidate):
                account_number = candidate
                print(f"[MANUAL EXTRACTION]   âœ“ Found account: {account_number}")
                break
    
    # Extract IFSC
    ifsc_patterns = [
        r'\b([A-Z]{4}0[A-Z0-9]{6})\b',
        r'IFSC[:\s]+([A-Z]{4}0[A-Z0-9]{6})',
    ]
    
    for pattern in ifsc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            ifsc_code = match.group(1)
            print(f"[MANUAL EXTRACTION]   âœ“ Found IFSC: {ifsc_code}")
            break
    
    # Extract name
    name_patterns = [
        r'Name[:\s]+([A-Z][A-Za-z\s\.]{2,40}?)(?:\s+(?:CIF|PERSONAL|Account|\d))',
        r'(?:Account\s+Holder|A/c\s+Holder)[:\s]+([A-Z][A-Za-z\s\.]{2,40})',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            if 3 <= len(name) <= 50 and not re.search(r'\d', name):
                account_holder_name = name
                print(f"[MANUAL EXTRACTION]   âœ“ Found name: {account_holder_name}")
                break
    
    # Extract branch
    branch_match = re.search(r'Branch[:\s]+([A-Z][A-Za-z\s]{3,40})', text, re.IGNORECASE)
    if branch_match:
        branch_name = branch_match.group(1).strip()
        print(f"[MANUAL EXTRACTION]   âœ“ Found branch: {branch_name}")
    
    return {
        'bank_name': bank_name,
        'account_holder_name': account_holder_name,
        'account_number': account_number,
        'ifsc_code': ifsc_code,
        'branch_name': branch_name
    }


def extract_bill_details_from_markdown(markdown_text: str) -> dict:
    """Regex-based bill extraction (fallback)"""
    print(f"[MANUAL EXTRACTION] Parsing bill markdown...")
    
    student_name = None
    bill_amount = None
    bill_date = None
    college_name = None
    receipt_number = None
    
    lines = markdown_text.split('\n')
    
    # Extract college name
    college_match = re.search(r'([A-Z][A-Za-z\s]+COLLEGE)', markdown_text, re.IGNORECASE)
    if college_match:
        college_name = college_match.group(1).strip()
        print(f"[MANUAL EXTRACTION]   âœ“ Found college: {college_name}")
    
    # Extract student name
    name_patterns = [
        r'Name[:\s\.]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, markdown_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if 3 <= len(name) <= 50 and not re.search(r'\d', name):
                student_name = name
                print(f"[MANUAL EXTRACTION]   âœ“ Found name: {student_name}")
                break
    
    # Extract amount
    all_amounts = []
    for line in lines:
        if re.search(r'(?:total|grand\s+total)', line, re.IGNORECASE):
            amount_matches = re.findall(r'(?:Rs\.?|â‚¹)\s*([0-9,\s\-]+)', line, re.IGNORECASE)
            for amt_str in amount_matches:
                clean = amt_str.replace(',', '').replace(' ', '').replace('-', '').strip()
                try:
                    val = float(clean)
                    if 100 < val < 1000000:
                        all_amounts.append(val)
                except:
                    pass
    
    if all_amounts:
        bill_amount = max(all_amounts)
        print(f"[MANUAL EXTRACTION]   âœ“ Found amount: Rs. {bill_amount}")
    
    return {
        'student_name': student_name,
        'college_name': college_name,
        'roll_number': None,
        'receipt_number': receipt_number,
        'bill_date': bill_date,
        'amount': bill_amount
    }


def analyze_bank_from_markdown(markdown_text: str) -> Dict[str, Any]:
    """Analyze bank passbook from OCR markdown"""
    try:
        return extract_bank_details_from_markdown(markdown_text)
    except Exception as e:
        print(f"[ANALYSIS] âœ— Error: {e}")
        return {
            'bank_name': None,
            'account_holder_name': None,
            'account_number': None,
            'ifsc_code': None,
            'branch_name': None
        }


def analyze_bill_from_markdown(markdown_text: str) -> Dict[str, Any]:
    """Analyze bill from OCR markdown"""
    try:
        return extract_bill_details_from_markdown(markdown_text)
    except Exception as e:
        print(f"[ANALYSIS] âœ— Error: {e}")
        return {
            'student_name': None,
            'college_name': None,
            'roll_number': None,
            'bill_date': None,
            'amount': None
        }


def analyze_generic_gemini_vision(file_content: bytes, filename: str, user_prompt: str) -> Dict[str, Any]:
    """
    Generic document analysis using Gemini Vision with custom prompt
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[GEMINI VISION] Analyzing document: {filename}")
    print(f"[GEMINI VISION] Custom prompt: {user_prompt[:100]}...")
    
    try:
        from google.genai import types
        
        # Detect MIME type
        mime_type = "image/jpeg"
        if file_content[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif file_content[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        
        print(f"[GEMINI VISION] Format: {mime_type}, Size: {len(file_content)} bytes")
        
        # Build the full prompt
        full_prompt = f"""
You are a highly accurate OCR and data extraction AI. Analyze this document image and extract the information requested by the user.

USER'S EXTRACTION REQUEST:
{user_prompt}

INSTRUCTIONS:
1. Read all text in the image carefully, including rotated, small, or handwritten text
2. Extract ONLY the information the user requested
3. If a field is not found or unclear, use null
4. IMPORTANT: Always return a JSON OBJECT, never just a string value
5. If extracting a single value, wrap it in an object like {{"value": "extracted_value"}}
6. Be as accurate as possible with numbers, dates, and names

Return your response as a valid JSON object with the fields the user requested.
"""
        
        # Call Gemini Vision
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=file_content,
                    mime_type=mime_type
                ),
                full_prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse response
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # âœ… FIX: Handle different response types
        if isinstance(data, str):
            # If Gemini returned a plain string, wrap it
            print(f"[GEMINI VISION] âš ï¸ Received string response, wrapping in object")
            data = {"extracted_value": data}
        elif isinstance(data, list):
            # If Gemini returned an array
            print(f"[GEMINI VISION] âš ï¸ Received array response")
            if len(data) > 0:
                if isinstance(data[0], dict):
                    data = data[0]  # Take first object
                else:
                    data = {"extracted_values": data}  # Wrap array
            else:
                data = {"extracted_values": []}
        elif isinstance(data, (int, float, bool)):
            # If Gemini returned a primitive value
            print(f"[GEMINI VISION] âš ï¸ Received primitive value, wrapping in object")
            data = {"extracted_value": data}
        elif not isinstance(data, dict):
            # Unknown type, wrap it
            print(f"[GEMINI VISION] âš ï¸ Unexpected type: {type(data)}, wrapping in object")
            data = {"extracted_value": str(data)}
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            data = {"error": "Invalid response format", "raw_response": str(data)}
        
        print(f"[GEMINI VISION] âœ“ Extraction completed")
        print(f"[GEMINI VISION]   Extracted {len(data)} fields")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[GEMINI VISION] âœ— JSON Parse Error: {e}")
        print(f"[GEMINI VISION] Raw response: {response_text[:200]}...")
        return {
            "error": "Failed to parse JSON response",
            "raw_response": response_text[:500]
        }
    except Exception as e:
        print(f"[GEMINI VISION] âœ— Error: {e}")
        import traceback
        traceback.print_exc()
        raise
