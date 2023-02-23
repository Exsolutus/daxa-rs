mod resource_id;
mod slot;

use proc_macro::TokenStream;
use syn;



#[proc_macro_derive(ResourceId)]
pub fn resource_id_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    //println!("{:?}", ast.data);

    // Build the trait implementation
    resource_id::impl_resource_id(&ast)
}

#[proc_macro_derive(Slot)]
pub fn slot_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    //println!("{:?}", ast.data);

    // Build the trait implementation
    slot::impl_slot(&ast)
}