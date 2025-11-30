def click_cards_and_extract_info_single_row(win, row_number: int = 1, collection_coords=None, card_dims=None, offset_accumulated=0) -> Dict[str, int]:
    """
    Process a single row of 6 cards - detects cards, clicks each one,
    captures the description zone, extracts card name and count, and returns summary.
    Returns dictionary with card names as keys and total counts as values.
    """
    print(f"[click_cards_and_extract_info_single_row] Starting row {row_number} card detection and clicking process...")

    # If collection_coords and card_dims are not provided, extract them from the window
    if collection_coords is None or card_dims is None:
        # Get window coordinates and handle negative coords
        left, top, width, height = win.left, win.top, win.width, win.height

        if left < 0 or top < 0:
            print(f"[click_cards_and_extract_info] Window at negative coords ({left},{top}). Attempting to move window to primary monitor (8,8).")
            try:
                win.moveTo(8, 8)
                time.sleep(0.35)
                left, top, width, height = win.left, win.top, win.width, win.height
                print(f"[click_cards_and_extract_info] Window moved to {left},{top} (size {width}x{height})")
            except Exception as e:
                print(f"[click_cards_and_extract_info] Could not move window: {e}. Proceeding with original coords.")

        # Grab full window screenshot
        try:
            full_window_img = grab_region((left, top, width, height))
            print(f"[click_cards_and_extract_info] Captured window screenshot: {full_window_img.shape}")
        except Exception as e:
            print(f"[click_cards_and_extract_info] Failed to grab window region: {e}")
            return {}

        # Load header template
        header_template_path = Path("templates/header.PNG")
        if not header_template_path.exists():
            print(f"[click_cards_and_extract_info] Header template not found at {header_template_path}")
            return {}

        header_template = cv2.imread(str(header_template_path))
        if header_template is None:
            print(f"[click_cards_and_extract_info] Failed to load header template from {header_template_path}")
            return {}

        print(f"[click_cards_and_extract_info] Loaded header template: {header_template.shape}")

        # Find header in window using template matching
        gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
        gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.5:
            print(f"[click_cards_and_extract_info] Header template match confidence too low: {max_val:.3f} (threshold: 0.5)")
            return {}

        header_x, header_y = max_loc
        header_h, header_w = gray_header.shape[:2]
        print(f"[click_cards_and_extract_info] Found header at window-relative position: ({header_x}, {header_y}) with confidence {max_val:.3f}")

        # Define card collection boundaries
        card_area_margin = int(header_w * 0.02)  # 2% margin from header edge to first card
        collection_area_x = header_x + card_area_margin
        collection_area_w = int(header_w * 0.935)  # Cards use ~93.5% of header width to include full 6th card

        # Define collection area
        collection_area_y = header_y + header_h + 10
        collection_area_h = height - collection_area_y - 50  # Extend to bottom of window (with 50px margin for UI)
        
        collection_coords = (collection_area_x, collection_area_y, collection_area_w, collection_area_h)
        
        # Calculate card dimensions for 5 rows total
        card_width = collection_area_w // 6  # Width of each card column
        card_height = collection_area_h // 5  # Height of each card row 
        card_dims = (card_width, card_height)
    else:
        # Extract values from provided parameters
        collection_area_x, collection_area_y, collection_area_w, collection_area_h = collection_coords
        card_width, card_height = card_dims
        # Get window coordinates for screen position calculations
        left, top, width, height = win.left, win.top, win.width, win.height

    # Create the appropriate directory structure based on row number
    if row_number <= 4:
        # Phase 1: rows 1-4 - need subdirectories for each row within phase1
        output_dir = Path(f"test_identifier/phase1/row_{row_number:02d}")
    else:
        # Phase 2: rows 5-12 organized in row_XX subdirectories under phase2
        output_dir = Path(f"test_identifier/phase2/row_{row_number:02d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate the detection area for this row based on phase
    if row_number <= 4:  # PHASE 1: Use offset approach
        # Calculate offset based on row number (20% of collection height per row)
        offset_per_row = int(collection_area_h * 0.20)  # 20% of total collection height
        row_offset = (row_number - 1) * offset_per_row  # row 1: offset=0, row 2: offset=20%, etc.
        
        # Calculate the target area for this row
        target_row_y = collection_area_y + row_offset
        target_row_h = card_height  # Height of a single row
        
        # Grab the specific row area from the full window screenshot
        full_window_img = grab_region((left, top, width, height))
        card_area_img = full_window_img[target_row_y:target_row_y + target_row_h, collection_area_x:collection_area_x + collection_area_w]
        
        # Save the row_full screenshot with boundaries
        row_full_img = card_area_img.copy()
        card_width_in_row = card_area_w // 6
        for i in range(1, 6):  # Draw 5 vertical lines to separate 6 cards
            line_x = i * card_width_in_row
            cv2.line(row_full_img, (line_x, 0), (line_x, target_row_h), (0, 255, 0), 2)  # Green lines

        # Save the row_full screenshot in the appropriate directory
        row_full_path = output_dir / f"{row_number}_row_full.png"
        cv2.imwrite(str(row_full_path), row_full_img)
        print(f"[click_cards_and_extract_info_single_row] Saved row {row_number} full screenshot with boundaries as {row_full_path}")
        
        # Calculate actual click area coordinates for this row
        actual_click_area_y = target_row_y
        actual_click_area_x = collection_area_x
        actual_click_area_w = collection_area_w
        actual_click_area_h = target_row_h
    else:  # PHASE 2: Use fixed area approach with accumulated offset
        # For all Phase 2 rows, use the same detection area (5th row position) with accumulated offset
        base_row_y = collection_area_y + 4 * card_height  # 5th row position
        actual_row_y = base_row_y + offset_accumulated  # Apply the accumulated offset from Phase 1
        
        # Grab the fixed detection area from the current window
        full_window_img = grab_region((left, top, width, height))
        card_area_img = full_window_img[actual_row_y:actual_row_y + card_height, collection_area_x:collection_area_x + collection_area_w]
        
        # Save the row_full screenshot with boundaries
        row_full_img = card_area_img.copy()
        card_width_in_row = card_area_w // 6
        for i in range(1, 6):  # Draw 5 vertical lines to separate 6 cards
            line_x = i * card_width_in_row
            cv2.line(row_full_img, (line_x, 0), (line_x, card_height), (0, 255, 0), 2)  # Green lines

        # Save the row_full screenshot in the appropriate directory
        row_full_path = output_dir / f"N_row_full.png"
        cv2.imwrite(str(row_full_path), row_full_img)
        print(f"[click_cards_and_extract_info_single_row] Saved row {row_number} full screenshot with boundaries as {row_full_path}")
        
        # Calculate actual click area coordinates for this row
        actual_click_area_y = actual_row_y
        actual_click_area_x = collection_area_x
        actual_click_area_w = collection_area_w
        actual_click_area_h = card_height

    # Calculate card positions in the detection area
    area_h, area_w = card_area_img.shape[:2]
    card_width_fixed = area_w // 6  # Divide area width by 6 cards
    card_height_fixed = min(area_h, int(card_width_fixed * 1.4))  # Card aspect ratio approximately 1:1.4

    # Start from top of area, center cards vertically if needed
    start_y = max(0, (area_h - card_height_fixed) // 2)

    cards_to_process = []

    for i in range(6):
        # Calculate card position for each of the 6 partition boxes
        card_x = i * card_width_fixed
        card_y = start_y

        # Add small margins to ensure we don't cut card borders
        margin_x = int(card_width_fixed * 0.05)  # 5% margin
        margin_y = int(card_height_fixed * 0.05)

        # Adjust boundaries to include margins but stay within bounds
        final_x = max(0, card_x + margin_x)
        final_y = max(0, card_y + margin_y)
        final_w = min(area_w - final_x, card_width_fixed - 2 * margin_x)
        final_h = min(area_h - final_y, card_height_fixed - 2 * margin_y)

        if final_w > 10 and final_h > 10:  # Ensure minimum size
            # Extract card image
            card_img = card_area_img[final_y:final_y + final_h, final_x:final_x + final_w].copy()

            # Save individual card for debugging in the appropriate directory
            card_path = output_dir / f"card_{i+1:02d}.png"
            cv2.imwrite(str(card_path), card_img)
            print(f"[click_cards_and_extract_info_single_row] Saved detected card {i+1} as {card_path}")

            # Store card with its bounding box (relative to card area)
            cards_to_process.append((card_img, (final_x, final_y, final_w, final_h)))
            print(f"[click_cards_and_extract_info] Card {i+1}: position=({final_x},{final_y}), size=({final_w}x{final_h})")
        else:
            print(f"[click_cards_and_extract_info] Warning: Card {i+1} has invalid dimensions: {final_w}x{final_h}")

    print(f"[click_cards_and_extract_info] Successfully detected {len(cards_to_process)} cards in row {row_number}")

    # Process each card by clicking and extracting info
    card_summary = {}  # Dictionary to store card name -> total count

    for i, (card_img, card_bbox) in enumerate(cards_to_process):
        print(f"\n[click_cards_and_extract_info] Processing card {i+1}/{len(cards_to_process)}")

        # Calculate click position (center of card relative to screen)
        card_x, card_y, card_w, card_h = card_bbox

        # Proper click position calculation for the partition box
        # Add window position + actual click area position + card position within area
        click_x = left + actual_click_area_x + card_x + card_w // 2
        click_y = top + actual_click_area_y + card_y + card_h // 2

        print(f"[click_cards_and_extract_info] Card {i+1} calculation:")
        print(f"  Window: ({left}, {top})")
        print(f"  Click area offset: ({actual_click_area_x}, {actual_click_area_y})")
        print(f"  Card bbox: ({card_x}, {card_y}, {card_w}, {card_h})")
        print(f"  Final click position: ({click_x}, {click_y})")

        try:
            # Click on the card in the partition box
            pyautogui.click(click_x, click_y)
            time.sleep(0.8)  # Wait for description to appear

            # Capture new screenshot after clicking
            clicked_window_img = grab_region((left, top, width, height))

            # Detect and capture description zone
            desc_zone_img = detect_and_capture_description_zone(clicked_window_img)

            if desc_zone_img is not None:
                # Save description zone in the appropriate directory
                desc_path = output_dir / f"desc_{i+1:02d}.png"
                cv2.imwrite(str(desc_path), desc_zone_img)
                print(f"[click_cards_and_extract_info_single_row] Saved description zone {i+1} as {desc_path}")
                
                # Extract card name and count from description zone
                card_name, count = ocr_description_zone_card_info(desc_zone_img, card_number=i+1, row_number=row_number)

                # Re-implement title_XX screenshot extraction from the OCR function
                # Since the OCR function already saves title images, no additional code needed here
                
                if card_name:  # Only process if we got a card name
                    # Aggregate counts for duplicate card names
                    if card_name in card_summary:
                        card_summary[card_name] += count
                        print(f"[click_cards_and_extract_info] Updated existing card '{card_name}': now {card_summary[card_name]} total")
                    else:
                        card_summary[card_name] = count
                        print(f"[click_cards_and_extract_info] New card '{card_name}': {count}")
                else:
                    print(f"[click_cards_and_extract_info] Could not extract card name from description zone")
            else:
                print(f"[click_cards_and_extract_info] Could not detect description zone for card {i+1}")

        except Exception as e:
            print(f"[click_cards_and_extract_info] Error processing card {i+1}: {e}")
            continue

    print(f"\n[click_cards_and_extract_info_single_row] Completed processing {len(cards_to_process)} cards for row {row_number}")
    return card_summary